from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests


@dataclass(slots=True)
class EmbeddingClient:
    """외부 임베딩 API 호출 전담 클라이언트."""

    api_url: str
    api_key: str | None = None
    model: str | None = None
    timeout: float = 10.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "EmbeddingClient":
        """input: cfg, output: client, desc: 설정 dict 기반 EmbeddingClient 생성."""
        api_url = cfg.get("api_url")
        if not api_url:
            raise ValueError("embedding_model.api_url이 필요합니다.")
        return cls(
            api_url=str(api_url),
            api_key=cfg.get("api_key"),
            model=cfg.get("model"),
            timeout=float(cfg.get("timeout", 10.0)),
        )

    def embed_text(self, text: str) -> np.ndarray:
        """input: text, output: np.ndarray, desc: 단건 텍스트 임베딩 API 호출."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {"input": text}
        if self.model:
            payload["model"] = self.model

        response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        body = response.json()

        if isinstance(body, dict):
            if "embedding" in body and isinstance(body["embedding"], list):
                return np.asarray(body["embedding"], dtype=np.float32)
            if "data" in body and body["data"]:
                first = body["data"][0]
                if isinstance(first, dict) and "embedding" in first:
                    return np.asarray(first["embedding"], dtype=np.float32)

        raise ValueError("Embedding API 응답에서 embedding 벡터를 찾지 못했습니다.")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """input: texts, output: np.ndarray, desc: 텍스트 리스트 임베딩."""
        return np.asarray([self.embed_text(t) for t in texts], dtype=np.float32)


class search_store:
    """ID 기준 Mongo CRUD + 벌크 load/save + 임베딩 인덱스 관리."""

    def __init__(self, embedding_client: EmbeddingClient, id_key: str = "contentId") -> None:
        """input: embedding_client/id_key, output: None, desc: store 상태 초기화."""
        self.embedding_client = embedding_client
        self.id_key = id_key

        self.contents: list[dict[str, Any]] = []
        self.content_ids: list[str] = []

        self.concat_texts: list[str] = []
        self.concat_embeddings: np.ndarray | None = None
        self.concat_embeddings_norm: np.ndarray | None = None
        self.concat_keys_used: list[str] = []

        self.field_embeddings: dict[str, np.ndarray] = {}
        self.field_embeddings_norm: dict[str, np.ndarray] = {}

    # ---------- bulk load/save ----------
    def replace_contents(self, items: list[dict[str, Any]]) -> None:
        """input: items, output: None, desc: contents 교체 + id 목록 동기화."""
        if any(not isinstance(item, dict) for item in items):
            raise ValueError("contents는 List[dict] 형태여야 합니다.")
        self.contents = items
        self.content_ids = [str(item[self.id_key]) for item in items if self.id_key in item]

    def load_contents_from_json(self, json_path: str | Path) -> None:
        """input: json_path, output: None, desc: JSON에서 content 벌크 로드."""
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON 루트는 List[dict] 형태여야 합니다.")
        self.replace_contents(data)

    def load_contents_from_mongodb(
        self,
        collection: Any,
        query: dict[str, Any] | None = None,
        limit: int | None = None,
        load_embeddings_if_present: bool = True,
        embedding_field: str = "embeddings",
    ) -> None:
        """input: collection/query..., output: None, desc: Mongo 벌크 로드 + 임베딩 캐시 복원."""
        cursor = collection.find(query or {})
        if limit:
            cursor = cursor.limit(limit)

        rows: list[dict[str, Any]] = []
        for row in cursor:
            row = dict(row)
            row.pop("_id", None)
            rows.append(row)

        self.replace_contents(rows)
        if load_embeddings_if_present:
            self.load_cached_embeddings_from_contents(embedding_field=embedding_field)

    def save_contents_to_mongodb(self, collection: Any, upsert: bool = True) -> None:
        """input: collection/upsert, output: None, desc: 현재 contents 벌크 저장."""
        if not self.contents:
            return

        if upsert:
            from pymongo import UpdateOne

            ops = []
            for item in self.contents:
                if self.id_key not in item:
                    raise KeyError(f"'{self.id_key}'가 없는 데이터는 저장할 수 없습니다.")
                ops.append(UpdateOne({self.id_key: item[self.id_key]}, {"$set": item}, upsert=True))
            if ops:
                collection.bulk_write(ops, ordered=False)
            return

        collection.insert_many(self.contents)

    # ---------- CRUD by id ----------
    def create_content(self, collection: Any, content: dict[str, Any], upsert: bool = False) -> None:
        """input: collection/content/upsert, output: None, desc: id 기준 단건 생성/upsert."""
        if self.id_key not in content:
            raise KeyError(f"'{self.id_key}'가 필요합니다.")
        if upsert:
            collection.update_one({self.id_key: content[self.id_key]}, {"$set": content}, upsert=True)
        else:
            collection.insert_one(content)

    def read_content(self, collection: Any, content_id: str) -> dict[str, Any] | None:
        """input: collection/content_id, output: dict|None, desc: id 기준 단건 조회."""
        row = collection.find_one({self.id_key: content_id})
        if row is None:
            return None
        row = dict(row)
        row.pop("_id", None)
        return row

    def update_content(self, collection: Any, content_id: str, patch: dict[str, Any]) -> bool:
        """input: collection/content_id/patch, output: bool, desc: id 기준 단건 업데이트."""
        result = collection.update_one({self.id_key: content_id}, {"$set": patch})
        return result.matched_count > 0

    def delete_content(self, collection: Any, content_id: str) -> bool:
        """input: collection/content_id, output: bool, desc: id 기준 단건 삭제."""
        result = collection.delete_one({self.id_key: content_id})
        return result.deleted_count > 0

    # ---------- embedding ----------
    def build_embeddings(
        self,
        mode: str = "both",
        concat_keys: Iterable[str] | None = None,
        per_key_fields: Iterable[str] | None = None,
        reuse_cached_embeddings: bool = True,
        embedding_field: str = "embeddings",
    ) -> None:
        """input: mode/keys..., output: None, desc: 인메모리 임베딩 캐시 생성(기존 캐시 재사용 가능)."""
        if not self.contents:
            raise ValueError("contents가 비어 있습니다.")
        if mode not in {"concat", "per_key", "both"}:
            raise ValueError("mode는 'concat' | 'per_key' | 'both' 이어야 합니다.")

        if reuse_cached_embeddings:
            self.load_cached_embeddings_from_contents(embedding_field=embedding_field)

        all_keys = sorted(self._collect_content_keys(self.contents))
        use_concat_keys = list(concat_keys) if concat_keys else all_keys
        use_per_key_fields = list(per_key_fields) if per_key_fields else all_keys

        need_concat = mode in {"concat", "both"} and not self._is_concat_cache_ready()
        need_per_key = mode in {"per_key", "both"} and not self._is_per_key_cache_ready(use_per_key_fields)

        if need_concat:
            self.concat_texts = [self._build_concat_text(item, use_concat_keys) for item in self.contents]
            self.concat_embeddings = self.embedding_client.embed_texts(self.concat_texts)
            self.concat_embeddings_norm = self._compute_row_l2_norms(self.concat_embeddings)
            self.concat_keys_used = use_concat_keys

        if need_per_key:
            self.field_embeddings.clear()
            self.field_embeddings_norm.clear()
            for field in use_per_key_fields:
                texts = [self._stringify_value_for_embedding(item.get(field, "")) for item in self.contents]
                matrix = self.embedding_client.embed_texts(texts)
                self.field_embeddings[field] = matrix
                self.field_embeddings_norm[field] = self._compute_row_l2_norms(matrix)

        self.content_ids = [str(item[self.id_key]) for item in self.contents if self.id_key in item]

    def embed_content_by_id(
        self,
        collection: Any,
        content_id: str,
        concat_keys: Iterable[str] | None = None,
        per_key_fields: Iterable[str] | None = None,
        embedding_field: str = "embeddings",
    ) -> dict[str, Any]:
        """input: collection/id/keys..., output: payload, desc: 단건 임베딩 생성 후 문서 저장."""
        item = self.read_content(collection, content_id)
        if item is None:
            raise KeyError(f"content_id='{content_id}' 문서를 찾지 못했습니다.")

        all_keys = sorted(item.keys())
        use_concat_keys = list(concat_keys) if concat_keys else all_keys
        use_per_key_fields = list(per_key_fields) if per_key_fields else all_keys

        concat_text = self._build_concat_text(item, use_concat_keys)
        concat_vector = self.embedding_client.embed_text(concat_text).astype(np.float32).tolist()

        per_key: dict[str, list[float]] = {}
        for field in use_per_key_fields:
            vector = self.embedding_client.embed_text(self._stringify_value_for_embedding(item.get(field, "")))
            per_key[field] = vector.astype(np.float32).tolist()

        payload = {
            "concat": concat_vector,
            "concat_meta": {"keys": use_concat_keys},
            "per_key": per_key,
        }
        collection.update_one({self.id_key: content_id}, {"$set": {embedding_field: payload}})
        return payload

    def embed_contents_bulk(
        self,
        collection: Any,
        query: dict[str, Any] | None = None,
        limit: int | None = None,
        concat_keys: Iterable[str] | None = None,
        per_key_fields: Iterable[str] | None = None,
        embedding_field: str = "embeddings",
        skip_if_embedding_exists: bool = True,
    ) -> int:
        """input: collection/query/keys..., output: int, desc: 벌크 임베딩 후 문서 업데이트."""
        self.load_contents_from_mongodb(
            collection,
            query=query,
            limit=limit,
            load_embeddings_if_present=False,
            embedding_field=embedding_field,
        )
        if not self.contents:
            return 0

        work_items = self.contents if not skip_if_embedding_exists else [item for item in self.contents if embedding_field not in item]
        if not work_items:
            return 0

        all_keys = sorted(self._collect_content_keys(work_items))
        use_concat_keys = list(concat_keys) if concat_keys else all_keys
        use_per_key_fields = list(per_key_fields) if per_key_fields else all_keys

        from pymongo import UpdateOne

        ops = []
        for item in work_items:
            if self.id_key not in item:
                continue

            concat_text = self._build_concat_text(item, use_concat_keys)
            concat_vector = self.embedding_client.embed_text(concat_text).astype(np.float32).tolist()

            per_key_payload: dict[str, list[float]] = {}
            for field in use_per_key_fields:
                vector = self.embedding_client.embed_text(self._stringify_value_for_embedding(item.get(field, "")))
                per_key_payload[field] = vector.astype(np.float32).tolist()

            payload = {
                "concat": concat_vector,
                "concat_meta": {"keys": use_concat_keys},
                "per_key": per_key_payload,
            }
            ops.append(UpdateOne({self.id_key: item[self.id_key]}, {"$set": {embedding_field: payload}}))

        if ops:
            collection.bulk_write(ops, ordered=False)
        return len(ops)

    def load_cached_embeddings_from_contents(self, embedding_field: str = "embeddings") -> None:
        """input: embedding_field, output: None, desc: 문서 내 임베딩을 인메모리 캐시로 복원."""
        if not self.contents:
            return

        concat_rows: list[list[float]] = []
        concat_ready = True
        concat_keys_candidate: list[str] | None = None
        per_key_rows: dict[str, list[list[float]]] = {}

        for item in self.contents:
            embedded = item.get(embedding_field)
            if not isinstance(embedded, dict):
                concat_ready = False
                continue

            concat_vec = embedded.get("concat")
            if isinstance(concat_vec, list):
                concat_rows.append(concat_vec)
            else:
                concat_ready = False

            concat_meta = embedded.get("concat_meta")
            if isinstance(concat_meta, dict) and isinstance(concat_meta.get("keys"), list):
                current_keys = [str(k) for k in concat_meta["keys"]]
                if concat_keys_candidate is None:
                    concat_keys_candidate = current_keys
                elif concat_keys_candidate != current_keys:
                    concat_keys_candidate = []

            per_key = embedded.get("per_key", {})
            if isinstance(per_key, dict):
                for field, vec in per_key.items():
                    if isinstance(vec, list):
                        per_key_rows.setdefault(field, []).append(vec)

        if concat_ready and len(concat_rows) == len(self.contents):
            self.concat_embeddings = np.asarray(concat_rows, dtype=np.float32)
            self.concat_embeddings_norm = self._compute_row_l2_norms(self.concat_embeddings)
            self.concat_keys_used = concat_keys_candidate or []

        self.field_embeddings.clear()
        self.field_embeddings_norm.clear()
        for field, rows in per_key_rows.items():
            if len(rows) != len(self.contents):
                continue
            matrix = np.asarray(rows, dtype=np.float32)
            self.field_embeddings[field] = matrix
            self.field_embeddings_norm[field] = self._compute_row_l2_norms(matrix)

    # ---------- internal ----------
    def _is_concat_cache_ready(self) -> bool:
        """input: -, output: bool, desc: concat 캐시 완전성 검증."""
        return (
            self.concat_embeddings is not None
            and self.concat_embeddings_norm is not None
            and len(self.concat_embeddings) == len(self.contents)
            and len(self.concat_embeddings_norm) == len(self.contents)
        )

    def _is_per_key_cache_ready(self, required_fields: list[str]) -> bool:
        """input: required_fields, output: bool, desc: per-key 캐시 완전성 검증."""
        if not required_fields:
            return False
        n = len(self.contents)
        for field in required_fields:
            matrix = self.field_embeddings.get(field)
            norms = self.field_embeddings_norm.get(field)
            if matrix is None or norms is None:
                return False
            if len(matrix) != n or len(norms) != n:
                return False
        return True

    @staticmethod
    def _collect_content_keys(items: list[dict[str, Any]]) -> set[str]:
        """input: items, output: set[str], desc: 문서 key 합집합 수집."""
        keys: set[str] = set()
        for item in items:
            keys.update(item.keys())
        return keys

    @staticmethod
    def _stringify_value_for_embedding(value: Any) -> str:
        """input: value, output: str, desc: 임베딩 입력용 문자열 변환."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _build_concat_text(self, item: dict[str, Any], keys: Iterable[str]) -> str:
        """input: item/keys, output: str, desc: 지정 key 값 concat 텍스트 생성."""
        return " ".join(self._stringify_value_for_embedding(item.get(k, "")) for k in keys).strip()

    @staticmethod
    def _compute_row_l2_norms(matrix: np.ndarray) -> np.ndarray:
        """input: matrix, output: norms, desc: 행 기준 L2 norm 계산."""
        return np.linalg.norm(matrix, axis=1) + 1e-12


class search_engine:
    """search_store 인덱스를 활용한 코사인 유사도 검색 엔진."""

    def __init__(self, embedding_mode: str, store: search_store, embedding_client: EmbeddingClient) -> None:
        """input: mode/store/client, output: None, desc: 검색 엔진 초기화."""
        if embedding_mode not in {"concat", "per_key"}:
            raise ValueError("embedding_mode는 'concat' 또는 'per_key'여야 합니다.")
        self.embedding_mode = embedding_mode
        self.search_store = store
        self.embedding_client = embedding_client

    @classmethod
    def from_config(cls, config_path: str | Path, store: search_store) -> "search_engine":
        """input: config_path/store, output: engine, desc: 설정 파일 기반 엔진 생성."""
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        mode = config.get("embedding_mode", "concat")
        client = EmbeddingClient.from_config(config.get("embedding_model", {}))
        return cls(mode, store, client)

    @staticmethod
    def cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray, matrix_norm: np.ndarray) -> np.ndarray:
        """input: query/matrix/norm, output: scores, desc: 코사인 유사도 계산."""
        query_norm = float(np.linalg.norm(query_vector) + 1e-12)
        return (matrix @ query_vector) / (matrix_norm * query_norm)

    def search_concat(self, keyword: str, top_k: int = 10) -> list[dict[str, Any]]:
        """input: keyword/top_k, output: list[dict], desc: concat 임베딩 기반 검색."""
        if self.search_store.concat_embeddings is None or self.search_store.concat_embeddings_norm is None:
            raise ValueError("concat 임베딩이 없습니다. build_embeddings(mode='concat' or 'both')를 먼저 호출하세요.")

        q = self.embedding_client.embed_text(keyword)
        scores = self.cosine_similarity(q, self.search_store.concat_embeddings, self.search_store.concat_embeddings_norm)
        return self._select_top_k_results(scores, top_k)

    def search_multi_weighted(
        self,
        keyword_by_field: dict[str, str],
        weight_by_field: dict[str, float] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """input: keyword_by_field/weight/top_k, output: list[dict], desc: per-key 가중 검색."""
        if not self.search_store.field_embeddings:
            raise ValueError("per_key 임베딩이 없습니다. build_embeddings(mode='per_key' or 'both')를 먼저 호출하세요.")

        weight_by_field = weight_by_field or {}
        final_scores: np.ndarray | None = None
        weight_sum = 0.0

        for field, keyword in keyword_by_field.items():
            matrix = self.search_store.field_embeddings.get(field)
            norms = self.search_store.field_embeddings_norm.get(field)
            if matrix is None or norms is None:
                continue

            weight = float(weight_by_field.get(field, 1.0))
            if weight <= 0:
                continue

            q = self.embedding_client.embed_text(keyword)
            field_scores = self.cosine_similarity(q, matrix, norms)
            final_scores = field_scores * weight if final_scores is None else final_scores + (field_scores * weight)
            weight_sum += weight

        if final_scores is None or weight_sum == 0:
            raise ValueError("유효한 keyword/weight 조합이 없습니다.")
        return self._select_top_k_results(final_scores / weight_sum, top_k)

    def search(
        self,
        keyword: str | None = None,
        keyword_by_field: dict[str, str] | None = None,
        weight_by_field: dict[str, float] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """input: keyword/keyword_by_field/weights/top_k, output: list[dict], desc: 모드별 검색."""
        if self.embedding_mode == "concat":
            if not keyword:
                raise ValueError("concat 모드에서는 keyword가 필요합니다.")
            return self.search_concat(keyword=keyword, top_k=top_k)

        if not keyword_by_field:
            raise ValueError("per_key 모드에서는 keyword_by_field가 필요합니다.")
        return self.search_multi_weighted(keyword_by_field, weight_by_field, top_k)

    def _select_top_k_results(self, scores: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        """input: scores/top_k, output: list[dict], desc: 상위 k개 결과 반환."""
        n = len(self.search_store.contents)
        if n == 0:
            return []

        k = max(1, min(top_k, n))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        return [{**self.search_store.contents[i], "score": float(scores[i])} for i in idx.tolist()]
