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
        vectors = [self.embed_text(t) for t in texts]
        return np.asarray(vectors, dtype=np.float32)


class search_store:
    """ID 기준 Mongo CRUD + 벌크 load/save + 임베딩 인덱스 관리."""

    def __init__(self, embedding_client: EmbeddingClient, id_key: str = "contentId") -> None:
        self.embedding_client = embedding_client
        self.id_key = id_key

        self.contents: list[dict[str, Any]] = []
        self.content_ids: list[str] = []

        self.concat_texts: list[str] = []
        self.concat_embeddings: np.ndarray | None = None
        self.concat_embeddings_norm: np.ndarray | None = None
        self.field_embeddings: dict[str, np.ndarray] = {}
        self.field_embeddings_norm: dict[str, np.ndarray] = {}

    # ---------- bulk load/save ----------
    def set_contents(self, items: list[dict[str, Any]]) -> None:
        if any(not isinstance(item, dict) for item in items):
            raise ValueError("contents는 List[dict] 형태여야 합니다.")
        self.contents = items
        self.content_ids = [str(item[self.id_key]) for item in items if self.id_key in item]

    def load_contents_from_json(self, json_path: str | Path) -> None:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON 루트는 List[dict] 형태여야 합니다.")
        self.set_contents(data)

    def load_contents_from_mongodb(self, collection: Any, query: dict[str, Any] | None = None, limit: int | None = None) -> None:
        cursor = collection.find(query or {})
        if limit:
            cursor = cursor.limit(limit)

        rows: list[dict[str, Any]] = []
        for row in cursor:
            row = dict(row)
            row.pop("_id", None)
            rows.append(row)
        self.set_contents(rows)

    def save_contents_to_mongodb(self, collection: Any, upsert: bool = True) -> None:
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
        if self.id_key not in content:
            raise KeyError(f"'{self.id_key}'가 필요합니다.")
        if upsert:
            collection.update_one({self.id_key: content[self.id_key]}, {"$set": content}, upsert=True)
            return
        collection.insert_one(content)

    def read_content(self, collection: Any, content_id: str) -> dict[str, Any] | None:
        row = collection.find_one({self.id_key: content_id})
        if row is None:
            return None
        row = dict(row)
        row.pop("_id", None)
        return row

    def update_content(self, collection: Any, content_id: str, patch: dict[str, Any]) -> bool:
        result = collection.update_one({self.id_key: content_id}, {"$set": patch})
        return result.matched_count > 0

    def delete_content(self, collection: Any, content_id: str) -> bool:
        result = collection.delete_one({self.id_key: content_id})
        return result.deleted_count > 0

    # ---------- embedding ----------
    def build_embeddings(
        self,
        mode: str = "both",
        concat_keys: Iterable[str] | None = None,
        per_key_fields: Iterable[str] | None = None,
    ) -> None:
        if not self.contents:
            raise ValueError("contents가 비어 있습니다.")
        if mode not in {"concat", "per_key", "both"}:
            raise ValueError("mode는 'concat' | 'per_key' | 'both' 이어야 합니다.")

        all_keys = sorted(self._all_keys())

        if mode in {"concat", "both"}:
            keys = list(concat_keys) if concat_keys else all_keys
            self.concat_texts = [self._concat_text(item, keys) for item in self.contents]
            self.concat_embeddings = self.embedding_client.embed_texts(self.concat_texts)
            self.concat_embeddings_norm = self._row_norm(self.concat_embeddings)

        if mode in {"per_key", "both"}:
            fields = list(per_key_fields) if per_key_fields else all_keys
            self.field_embeddings.clear()
            self.field_embeddings_norm.clear()
            for field in fields:
                texts = [self._to_text(item.get(field, "")) for item in self.contents]
                matrix = self.embedding_client.embed_texts(texts)
                self.field_embeddings[field] = matrix
                self.field_embeddings_norm[field] = self._row_norm(matrix)

        self.content_ids = [str(item[self.id_key]) for item in self.contents if self.id_key in item]

    def embed_content_by_id(
        self,
        collection: Any,
        content_id: str,
        concat_keys: Iterable[str] | None = None,
        per_key_fields: Iterable[str] | None = None,
        embedding_field: str = "embeddings",
    ) -> dict[str, Any]:
        item = self.read_content(collection, content_id)
        if item is None:
            raise KeyError(f"content_id='{content_id}' 문서를 찾지 못했습니다.")

        all_keys = sorted(item.keys())
        concat_keys = list(concat_keys) if concat_keys else all_keys
        per_key_fields = list(per_key_fields) if per_key_fields else all_keys

        concat_text = self._concat_text(item, concat_keys)
        concat_vector = self.embedding_client.embed_text(concat_text).astype(np.float32).tolist()

        per_key: dict[str, list[float]] = {}
        for field in per_key_fields:
            vector = self.embedding_client.embed_text(self._to_text(item.get(field, "")))
            per_key[field] = vector.astype(np.float32).tolist()

        payload = {"concat": concat_vector, "per_key": per_key}
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
    ) -> int:
        self.load_contents_from_mongodb(collection, query=query, limit=limit)
        if not self.contents:
            return 0

        self.build_embeddings(mode="both", concat_keys=concat_keys, per_key_fields=per_key_fields)

        from pymongo import UpdateOne

        ops = []
        for idx, item in enumerate(self.contents):
            if self.id_key not in item:
                continue
            per_key_payload = {
                field: self.field_embeddings[field][idx].astype(np.float32).tolist()
                for field in self.field_embeddings
            }
            payload = {
                "concat": self.concat_embeddings[idx].astype(np.float32).tolist() if self.concat_embeddings is not None else None,
                "per_key": per_key_payload,
            }
            ops.append(UpdateOne({self.id_key: item[self.id_key]}, {"$set": {embedding_field: payload}}))

        if ops:
            collection.bulk_write(ops, ordered=False)
        return len(ops)

    # ---------- internal ----------
    def _all_keys(self) -> set[str]:
        keys: set[str] = set()
        for item in self.contents:
            keys.update(item.keys())
        return keys

    @staticmethod
    def _to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _concat_text(self, item: dict[str, Any], keys: Iterable[str]) -> str:
        return " ".join(self._to_text(item.get(k, "")) for k in keys).strip()

    @staticmethod
    def _row_norm(matrix: np.ndarray) -> np.ndarray:
        return np.linalg.norm(matrix, axis=1) + 1e-12


class search_engine:
    """search_store 인덱스를 활용한 코사인 유사도 검색 엔진."""

    def __init__(self, embedding_mode: str, store: search_store, embedding_client: EmbeddingClient) -> None:
        if embedding_mode not in {"concat", "per_key"}:
            raise ValueError("embedding_mode는 'concat' 또는 'per_key'여야 합니다.")
        self.embedding_mode = embedding_mode
        self.search_store = store
        self.embedding_client = embedding_client

    @classmethod
    def from_config(cls, config_path: str | Path, store: search_store) -> "search_engine":
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        mode = config.get("embedding_mode", "concat")
        client = EmbeddingClient.from_config(config.get("embedding_model", {}))
        return cls(mode, store, client)

    @staticmethod
    def cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray, matrix_norm: np.ndarray) -> np.ndarray:
        query_norm = float(np.linalg.norm(query_vector) + 1e-12)
        return (matrix @ query_vector) / (matrix_norm * query_norm)

    def search_concat(self, keyword: str, top_k: int = 10) -> list[dict[str, Any]]:
        if self.search_store.concat_embeddings is None or self.search_store.concat_embeddings_norm is None:
            raise ValueError("concat 임베딩이 없습니다. build_embeddings(mode='concat' or 'both')를 먼저 호출하세요.")
        q = self.embedding_client.embed_text(keyword)
        scores = self.cosine_similarity(q, self.search_store.concat_embeddings, self.search_store.concat_embeddings_norm)
        return self._top_k(scores, top_k)

    def search_multi_weighted(
        self,
        keyword_by_field: dict[str, str],
        weight_by_field: dict[str, float] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
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
        return self._top_k(final_scores / weight_sum, top_k)

    def search(
        self,
        keyword: str | None = None,
        keyword_by_field: dict[str, str] | None = None,
        weight_by_field: dict[str, float] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        if self.embedding_mode == "concat":
            if not keyword:
                raise ValueError("concat 모드에서는 keyword가 필요합니다.")
            return self.search_concat(keyword=keyword, top_k=top_k)

        if not keyword_by_field:
            raise ValueError("per_key 모드에서는 keyword_by_field가 필요합니다.")
        return self.search_multi_weighted(keyword_by_field, weight_by_field, top_k)

    def _top_k(self, scores: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        n = len(self.search_store.contents)
        if n == 0:
            return []

        k = max(1, min(top_k, n))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        return [{**self.search_store.contents[i], "score": float(scores[i])} for i in idx.tolist()]
