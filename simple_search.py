from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests


REQUIRED_CONTENT_KEYS = (
    "contentId",
    "contentNm",
    "contentDesc",
    "contentGC",
    "contentCusGC",
    "contentPeriod",
    "contentObject",
    "contentAction",
    "contentMetric",
)


@dataclass(slots=True)
class _EmbeddingConfig:
    """Embedding 호출 설정 DTO."""

    api_url: str
    api_key: str | None
    model: str | None
    timeout: float
    provider: str
    mock_dimension: int


class search_store:
    """검색 대상 저장소 + 임베딩 인덱스 관리 클래스."""

    def __init__(self, embedding_model: dict[str, Any]) -> None:
        """in: embedding_model(dict), out: None, desc: 저장소/인덱스 상태 초기화."""
        self.contents: list[dict[str, Any]] = []
        self.embedding_model: dict[str, Any] = embedding_model

        self.concat_texts: list[str] = []
        self.concat_embeddings: np.ndarray | None = None
        self.concat_embeddings_norm: np.ndarray | None = None

        self.field_embeddings: dict[str, np.ndarray] = {}
        self.field_embeddings_norm: dict[str, np.ndarray] = {}
        self._content_ids: list[str] = []

    def load_from_json(self, json_path: str | Path) -> None:
        """in: json_path, out: None, desc: JSON 로드 + 필수 키 검증."""
        path = Path(json_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON 루트는 List[dict] 형태여야 합니다.")
        self._set_contents(data)

    def load_from_mongodb(
        self,
        mongo_client: Any,
        db_name: str,
        collection_name: str,
        query: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> None:
        """in: mongo_client/db/collection/query/limit, out: None, desc: 단일 collection에서 contents 로드."""
        query = query or {}
        collection = mongo_client[db_name][collection_name]

        cursor = collection.find(query)
        if limit:
            cursor = cursor.limit(limit)

        items = [self._normalize_mongo_row(row) for row in cursor]
        self._set_contents(items)

    def load_from_mongodb_split(
        self,
        mongo_client: Any,
        db_name: str,
        metadata_collection_name: str,
        embedding_collection_name: str,
        query: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> None:
        """in: 메타/임베딩 collection, out: None, desc: 메타와 임베딩을 contentId 기준으로 나눠 로드."""
        query = query or {}
        metadata_collection = mongo_client[db_name][metadata_collection_name]
        embedding_collection = mongo_client[db_name][embedding_collection_name]

        cursor = metadata_collection.find(query)
        if limit:
            cursor = cursor.limit(limit)

        metadata_items = [self._normalize_mongo_row(row) for row in cursor]
        self._set_contents(metadata_items)

        content_ids = [str(item["contentId"]) for item in self.contents]
        if not content_ids:
            return

        embed_docs = embedding_collection.find({"contentId": {"$in": content_ids}})
        doc_by_id = {str(doc.get("contentId")): doc for doc in embed_docs}

        concat_vectors: list[list[float]] = []
        concat_ready = True
        per_key_vectors: dict[str, list[list[float]]] = {}

        for cid in content_ids:
            doc = doc_by_id.get(cid)
            if not doc:
                concat_ready = False
                continue

            concat_vector = doc.get("concat_embedding")
            if isinstance(concat_vector, list):
                concat_vectors.append(concat_vector)
            else:
                concat_ready = False

            field_embeddings = doc.get("field_embeddings", {})
            if isinstance(field_embeddings, dict):
                for field, vector in field_embeddings.items():
                    if isinstance(vector, list):
                        per_key_vectors.setdefault(field, []).append(vector)

        if concat_ready and len(concat_vectors) == len(content_ids):
            self.concat_embeddings = np.asarray(concat_vectors, dtype=np.float32)
            self.concat_embeddings_norm = self._row_norm(self.concat_embeddings)

        self.field_embeddings.clear()
        self.field_embeddings_norm.clear()
        for field, vectors in per_key_vectors.items():
            if len(vectors) != len(content_ids):
                continue
            matrix = np.asarray(vectors, dtype=np.float32)
            self.field_embeddings[field] = matrix
            self.field_embeddings_norm[field] = self._row_norm(matrix)

        self._content_ids = content_ids

    def save_contents_to_mongodb(
        self,
        mongo_client: Any,
        db_name: str,
        collection_name: str,
        replace_by_content_id: bool = True,
    ) -> None:
        """in: mongo_client/db/collection, out: None, desc: 단일 collection에 contents 저장."""
        if not self.contents:
            return

        collection = mongo_client[db_name][collection_name]

        if replace_by_content_id:
            ops = [self._build_upsert_operation(content) for content in self.contents]
            collection.bulk_write(ops, ordered=False)
            return

        collection.insert_many(self.contents)

    def save_to_mongodb_split(
        self,
        mongo_client: Any,
        db_name: str,
        metadata_collection_name: str,
        embedding_collection_name: str,
        replace_by_content_id: bool = True,
    ) -> None:
        """in: 메타/임베딩 collection, out: None, desc: 필수 메타와 임베딩을 contentId 기준 분리 저장."""
        if not self.contents:
            return

        metadata_collection = mongo_client[db_name][metadata_collection_name]
        embedding_collection = mongo_client[db_name][embedding_collection_name]

        metadata_docs = [self._extract_required_metadata(content) for content in self.contents]
        embedding_docs = self._build_embedding_documents_for_split_storage()

        if replace_by_content_id:
            meta_ops = [self._build_upsert_operation(doc) for doc in metadata_docs]
            if meta_ops:
                metadata_collection.bulk_write(meta_ops, ordered=False)

            embed_ops = [self._build_upsert_operation(doc) for doc in embedding_docs]
            if embed_ops:
                embedding_collection.bulk_write(embed_ops, ordered=False)
            return

        metadata_collection.insert_many(metadata_docs)
        if embedding_docs:
            embedding_collection.insert_many(embedding_docs)

    def build_embeddings(
        self,
        mode: str = "both",
        concat_keys: Iterable[str] | None = None,
        per_key_fields: Iterable[str] | None = None,
    ) -> None:
        """in: mode/keys, out: None, desc: concat/per_key 임베딩과 norm 캐시 생성."""
        self._validate_before_embedding(mode)
        concat_keys = list(concat_keys or REQUIRED_CONTENT_KEYS)
        per_key_fields = list(per_key_fields or REQUIRED_CONTENT_KEYS)

        if mode in ("concat", "both"):
            self._build_concat_embeddings(concat_keys)

        if mode in ("per_key", "both"):
            self._build_per_key_embeddings(per_key_fields)

        self._content_ids = [str(c["contentId"]) for c in self.contents]

    def _set_contents(self, items: list[dict[str, Any]]) -> None:
        """in: items, out: None, desc: content 스키마 검증 후 self.contents 설정."""
        normalized: list[dict[str, Any]] = []
        for idx, content in enumerate(items):
            if not isinstance(content, dict):
                raise ValueError(f"index={idx} 데이터가 dict가 아닙니다.")
            missing = [k for k in REQUIRED_CONTENT_KEYS if k not in content]
            if missing:
                raise ValueError(f"index={idx} 데이터에 필수 key 누락: {missing}")
            normalized.append(content)

        self.contents = normalized

    @staticmethod
    def _normalize_mongo_row(row: dict[str, Any]) -> dict[str, Any]:
        """in: mongo row, out: dict, desc: _id 제거."""
        row = dict(row)
        row.pop("_id", None)
        return row

    @staticmethod
    def _extract_required_metadata(content: dict[str, Any]) -> dict[str, Any]:
        """in: content, out: metadata dict, desc: 필수 메타 key만 추출."""
        return {key: content.get(key) for key in REQUIRED_CONTENT_KEYS}

    def _build_embedding_documents_for_split_storage(self) -> list[dict[str, Any]]:
        """in: 없음, out: embedding docs, desc: contentId별 concat/per_key 임베딩 문서 생성."""
        docs: list[dict[str, Any]] = []
        n = len(self.contents)

        has_concat = self.concat_embeddings is not None and len(self.concat_embeddings) == n
        has_per_key = bool(self.field_embeddings)

        for i, content in enumerate(self.contents):
            doc: dict[str, Any] = {"contentId": str(content["contentId"])}

            if has_concat:
                doc["concat_embedding"] = self.concat_embeddings[i].astype(np.float32).tolist()

            if has_per_key:
                field_payload: dict[str, list[float]] = {}
                for field, matrix in self.field_embeddings.items():
                    if len(matrix) == n:
                        field_payload[field] = matrix[i].astype(np.float32).tolist()
                if field_payload:
                    doc["field_embeddings"] = field_payload

            docs.append(doc)

        return docs

    @staticmethod
    def _build_upsert_operation(content: dict[str, Any]) -> Any:
        """in: content, out: UpdateOne, desc: contentId 기반 upsert 연산 생성."""
        try:
            from pymongo import UpdateOne
        except ImportError as exc:
            raise ImportError("pymongo가 필요합니다. `pip install pymongo` 후 재시도하세요.") from exc

        return UpdateOne(
            {"contentId": content["contentId"]},
            {"$set": content},
            upsert=True,
        )

    def _validate_before_embedding(self, mode: str) -> None:
        """in: mode, out: None, desc: 모드/데이터 유효성 확인."""
        if mode not in ("concat", "per_key", "both"):
            raise ValueError("mode는 'concat', 'per_key', 'both' 중 하나여야 합니다.")
        if not self.contents:
            raise ValueError("contents가 비어 있습니다. 먼저 DB/JSON을 로드하세요.")

    def _build_concat_embeddings(self, concat_keys: list[str]) -> None:
        """in: concat_keys, out: None, desc: concat 임베딩/노름 생성."""
        self.concat_texts = [self._concat_content_text(c, concat_keys) for c in self.contents]
        self.concat_embeddings = self._embed_texts(self.concat_texts)
        self.concat_embeddings_norm = self._row_norm(self.concat_embeddings)

    def _build_per_key_embeddings(self, per_key_fields: list[str]) -> None:
        """in: per_key_fields, out: None, desc: 필드별 임베딩/노름 생성."""
        self.field_embeddings.clear()
        self.field_embeddings_norm.clear()

        for field in per_key_fields:
            texts = [self._to_text(c.get(field, "")) for c in self.contents]
            matrix = self._embed_texts(texts)
            self.field_embeddings[field] = matrix
            self.field_embeddings_norm[field] = self._row_norm(matrix)

    @staticmethod
    def _to_text(value: Any) -> str:
        """in: any, out: str, desc: 임베딩용 문자열 변환."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _concat_content_text(self, content: dict[str, Any], keys: Iterable[str]) -> str:
        """in: content/keys, out: str, desc: key값 결합 문자열 생성."""
        return " ".join(self._to_text(content.get(key, "")) for key in keys).strip()

    @staticmethod
    def _row_norm(matrix: np.ndarray) -> np.ndarray:
        """in: matrix(n,d), out: norm(n), desc: 행 L2 norm + epsilon."""
        return np.linalg.norm(matrix, axis=1) + 1e-12

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """in: texts, out: matrix, desc: 텍스트 리스트 임베딩."""
        vectors = [self._embed_single_text(t) for t in texts]
        return np.asarray(vectors, dtype=np.float32)

    def _embed_single_text(self, text: str) -> list[float]:
        """in: text, out: vector, desc: mock 또는 API 호출로 단건 임베딩."""
        cfg = self._parse_embedding_config(self.embedding_model)

        if cfg.provider == "mock" or cfg.api_url == "mock":
            return self._mock_embedding(text, cfg.mock_dimension)

        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

        payload = {"input": text}
        if cfg.model:
            payload["model"] = cfg.model

        response = requests.post(cfg.api_url, headers=headers, json=payload, timeout=cfg.timeout)
        response.raise_for_status()
        body = response.json()

        if isinstance(body, dict):
            if "embedding" in body and isinstance(body["embedding"], list):
                return body["embedding"]
            if "data" in body and body["data"]:
                first = body["data"][0]
                if isinstance(first, dict) and "embedding" in first:
                    return first["embedding"]

        raise ValueError("Embedding API 응답에서 embedding 벡터를 찾지 못했습니다.")

    @staticmethod
    def _mock_embedding(text: str, dim: int) -> list[float]:
        """in: text/dim, out: vector, desc: deterministic mock 임베딩."""
        digest = sha256(text.encode("utf-8")).digest()
        result = []
        for i in range(dim):
            b = digest[i % len(digest)]
            result.append((b / 255.0) * 2 - 1)
        return result

    @staticmethod
    def _parse_embedding_config(model_cfg: dict[str, Any]) -> _EmbeddingConfig:
        """in: model_cfg dict, out: _EmbeddingConfig, desc: 기본값 포함 파싱."""
        return _EmbeddingConfig(
            api_url=str(model_cfg.get("api_url", "mock")),
            api_key=model_cfg.get("api_key"),
            model=model_cfg.get("model"),
            timeout=float(model_cfg.get("timeout", 10.0)),
            provider=str(model_cfg.get("provider", "generic")),
            mock_dimension=int(model_cfg.get("mock_dimension", 64)),
        )


class search_engine:
    """search_store 기반 코사인 유사도 검색 엔진."""

    def __init__(
        self,
        embedding_mode: str,
        store: search_store,
        embedding_model: dict[str, Any],
    ) -> None:
        """in: mode/store/model, out: None, desc: 검색 엔진 초기화."""
        if embedding_mode not in {"concat", "per_key"}:
            raise ValueError("embedding_mode는 'concat' 또는 'per_key'여야 합니다.")
        self.embedding_mode = embedding_mode
        self.search_store = store
        self.embedding_model = embedding_model

    @classmethod
    def from_config(cls, config_path: str | Path, store: search_store) -> "search_engine":
        """in: config_path/store, out: engine, desc: 설정 파일 기반 엔진 생성."""
        path = Path(config_path)
        config = json.loads(path.read_text(encoding="utf-8"))
        mode = config.get("embedding_mode", "concat")
        embedding_model = config.get("embedding_model", {})
        return cls(embedding_mode=mode, store=store, embedding_model=embedding_model)

    def cosine_similarity(self, query_vector: np.ndarray, matrix: np.ndarray, matrix_norm: np.ndarray) -> np.ndarray:
        """in: query/matrix/norm, out: scores, desc: 코사인 유사도 계산."""
        query_norm = float(np.linalg.norm(query_vector) + 1e-12)
        dots = matrix @ query_vector
        return dots / (matrix_norm * query_norm)

    def search_concat(self, keyword: str, top_k: int = 10) -> list[dict[str, Any]]:
        """in: keyword/top_k, out: 결과 리스트, desc: concat 인덱스 검색."""
        self._validate_concat_ready()

        query_vec = np.asarray(self.search_store._embed_single_text(keyword), dtype=np.float32)
        scores = self.cosine_similarity(
            query_vector=query_vec,
            matrix=self.search_store.concat_embeddings,
            matrix_norm=self.search_store.concat_embeddings_norm,
        )
        return self._top_k(scores, top_k)

    def search_multi_weighted(
        self,
        keyword_by_field: dict[str, str],
        weight_by_field: dict[str, float] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """in: field-keyword/weight/top_k, out: 결과 리스트, desc: 필드 가중 검색."""
        self._validate_per_key_ready()
        weight_by_field = weight_by_field or {}

        final_scores: np.ndarray | None = None
        weight_sum = 0.0

        for field, keyword in keyword_by_field.items():
            if field not in self.search_store.field_embeddings:
                raise KeyError(f"field='{field}' 임베딩이 없습니다.")

            weight = float(weight_by_field.get(field, 1.0))
            if weight <= 0:
                continue

            query_vec = np.asarray(self.search_store._embed_single_text(keyword), dtype=np.float32)
            field_scores = self.cosine_similarity(
                query_vector=query_vec,
                matrix=self.search_store.field_embeddings[field],
                matrix_norm=self.search_store.field_embeddings_norm[field],
            )

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
        """in: keyword args, out: 결과 리스트, desc: 모드별 검색 분기."""
        if self.embedding_mode == "concat":
            if not keyword:
                raise ValueError("concat 모드에서는 keyword가 필요합니다.")
            return self.search_concat(keyword=keyword, top_k=top_k)

        if not keyword_by_field:
            raise ValueError("per_key 모드에서는 keyword_by_field가 필요합니다.")
        return self.search_multi_weighted(
            keyword_by_field=keyword_by_field,
            weight_by_field=weight_by_field,
            top_k=top_k,
        )

    def _validate_concat_ready(self) -> None:
        """in: -, out: None, desc: concat 임베딩 준비 상태 확인."""
        if self.search_store.concat_embeddings is None or self.search_store.concat_embeddings_norm is None:
            raise ValueError("concat 임베딩이 없습니다. search_store.build_embeddings(mode='concat' or 'both')를 먼저 호출하세요.")

    def _validate_per_key_ready(self) -> None:
        """in: -, out: None, desc: per_key 임베딩 준비 상태 확인."""
        if not self.search_store.field_embeddings:
            raise ValueError("per_key 임베딩이 없습니다. search_store.build_embeddings(mode='per_key' or 'both')를 먼저 호출하세요.")

    def _top_k(self, scores: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        """in: scores/top_k, out: 결과 리스트, desc: argpartition 기반 top-k 추출."""
        n = len(self.search_store.contents)
        k = max(1, min(top_k, n))

        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        result = []
        for i in idx.tolist():
            item = dict(self.search_store.contents[i])
            item["score"] = float(scores[i])
            result.append(item)
        return result
