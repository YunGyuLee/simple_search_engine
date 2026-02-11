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
    api_url: str
    api_key: str | None
    model: str | None
    timeout: float
    provider: str
    mock_dimension: int


class search_store:
    """검색 대상 content 저장소 + 임베딩 인덱스 관리 클래스."""

    def __init__(self, embedding_model: dict[str, Any]) -> None:
        self.contents: list[dict[str, Any]] = []
        self.embedding_model: dict[str, Any] = embedding_model

        # 검색 속도 최적화를 위해 numpy matrix 기반으로 저장
        self.concat_texts: list[str] = []
        self.concat_embeddings: np.ndarray | None = None
        self.concat_embeddings_norm: np.ndarray | None = None

        self.field_embeddings: dict[str, np.ndarray] = {}
        self.field_embeddings_norm: dict[str, np.ndarray] = {}
        self._content_ids: list[str] = []

    # -------------------------------
    # Data Load / Save
    # -------------------------------
    def load_from_json(self, json_path: str | Path) -> None:
        """JSON 파일에서 contents를 로드하고 검증 후 저장한다."""
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
        """외부에서 생성한 MongoClient를 받아 contents를 로드한다."""
        query = query or {}
        collection = mongo_client[db_name][collection_name]

        cursor = collection.find(query)
        if limit:
            cursor = cursor.limit(limit)

        rows = list(cursor)
        items = [self._normalize_mongo_row(row) for row in rows]
        self._set_contents(items)

    def save_contents_to_mongodb(
        self,
        mongo_client: Any,
        db_name: str,
        collection_name: str,
        replace_by_content_id: bool = True,
    ) -> None:
        """외부 MongoClient를 사용해 현재 contents를 MongoDB에 저장한다."""
        if not self.contents:
            return

        collection = mongo_client[db_name][collection_name]

        if replace_by_content_id:
            ops = [
                self._build_upsert_operation(content)
                for content in self.contents
            ]
            collection.bulk_write(ops, ordered=False)
            return

        collection.insert_many(self.contents)

    # -------------------------------
    # Embedding
    # -------------------------------
    def build_embeddings(
        self,
        mode: str = "both",
        concat_keys: Iterable[str] | None = None,
        per_key_fields: Iterable[str] | None = None,
    ) -> None:
        """
        mode:
          - concat: 여러 key concat 텍스트 임베딩
          - per_key: key별 개별 임베딩
          - both: 둘 다
        """
        self._validate_before_embedding(mode)
        concat_keys = list(concat_keys or REQUIRED_CONTENT_KEYS)
        per_key_fields = list(per_key_fields or REQUIRED_CONTENT_KEYS)

        if mode in ("concat", "both"):
            self._build_concat_embeddings(concat_keys)

        if mode in ("per_key", "both"):
            self._build_per_key_embeddings(per_key_fields)

        self._content_ids = [str(c["contentId"]) for c in self.contents]

    # -------------------------------
    # Internal helpers
    # -------------------------------
    def _set_contents(self, items: list[dict[str, Any]]) -> None:
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
        row = dict(row)
        row.pop("_id", None)
        return row

    @staticmethod
    def _build_upsert_operation(content: dict[str, Any]) -> Any:
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
        if mode not in ("concat", "per_key", "both"):
            raise ValueError("mode는 'concat', 'per_key', 'both' 중 하나여야 합니다.")
        if not self.contents:
            raise ValueError("contents가 비어 있습니다. 먼저 DB/JSON을 로드하세요.")

    def _build_concat_embeddings(self, concat_keys: list[str]) -> None:
        self.concat_texts = [self._concat_content_text(c, concat_keys) for c in self.contents]
        self.concat_embeddings = self._embed_texts(self.concat_texts)
        self.concat_embeddings_norm = self._row_norm(self.concat_embeddings)

    def _build_per_key_embeddings(self, per_key_fields: list[str]) -> None:
        self.field_embeddings.clear()
        self.field_embeddings_norm.clear()

        for field in per_key_fields:
            texts = [self._to_text(c.get(field, "")) for c in self.contents]
            matrix = self._embed_texts(texts)
            self.field_embeddings[field] = matrix
            self.field_embeddings_norm[field] = self._row_norm(matrix)

    @staticmethod
    def _to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _concat_content_text(self, content: dict[str, Any], keys: Iterable[str]) -> str:
        return " ".join(self._to_text(content.get(key, "")) for key in keys).strip()

    @staticmethod
    def _row_norm(matrix: np.ndarray) -> np.ndarray:
        return np.linalg.norm(matrix, axis=1) + 1e-12

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = [self._embed_single_text(t) for t in texts]
        return np.asarray(vectors, dtype=np.float32)

    def _embed_single_text(self, text: str) -> list[float]:
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
        digest = sha256(text.encode("utf-8")).digest()
        result = []
        for i in range(dim):
            b = digest[i % len(digest)]
            result.append((b / 255.0) * 2 - 1)
        return result

    @staticmethod
    def _parse_embedding_config(model_cfg: dict[str, Any]) -> _EmbeddingConfig:
        return _EmbeddingConfig(
            api_url=str(model_cfg.get("api_url", "mock")),
            api_key=model_cfg.get("api_key"),
            model=model_cfg.get("model"),
            timeout=float(model_cfg.get("timeout", 10.0)),
            provider=str(model_cfg.get("provider", "generic")),
            mock_dimension=int(model_cfg.get("mock_dimension", 64)),
        )


class search_engine:
    """search_store를 기반으로 코사인 유사도 검색을 수행하는 엔진 클래스."""

    def __init__(
        self,
        embedding_mode: str,
        store: search_store,
        embedding_model: dict[str, Any],
    ) -> None:
        if embedding_mode not in {"concat", "per_key"}:
            raise ValueError("embedding_mode는 'concat' 또는 'per_key'여야 합니다.")
        self.embedding_mode = embedding_mode
        self.search_store = store
        self.embedding_model = embedding_model

    @classmethod
    def from_config(cls, config_path: str | Path, store: search_store) -> "search_engine":
        path = Path(config_path)
        config = json.loads(path.read_text(encoding="utf-8"))
        mode = config.get("embedding_mode", "concat")
        embedding_model = config.get("embedding_model", {})
        return cls(embedding_mode=mode, store=store, embedding_model=embedding_model)

    def cosine_similarity(self, query_vector: np.ndarray, matrix: np.ndarray, matrix_norm: np.ndarray) -> np.ndarray:
        query_norm = float(np.linalg.norm(query_vector) + 1e-12)
        dots = matrix @ query_vector
        return dots / (matrix_norm * query_norm)

    def search_concat(self, keyword: str, top_k: int = 10) -> list[dict[str, Any]]:
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
        if self.search_store.concat_embeddings is None or self.search_store.concat_embeddings_norm is None:
            raise ValueError("concat 임베딩이 없습니다. search_store.build_embeddings(mode='concat' or 'both')를 먼저 호출하세요.")

    def _validate_per_key_ready(self) -> None:
        if not self.search_store.field_embeddings:
            raise ValueError("per_key 임베딩이 없습니다. search_store.build_embeddings(mode='per_key' or 'both')를 먼저 호출하세요.")

    def _top_k(self, scores: np.ndarray, top_k: int) -> list[dict[str, Any]]:
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
