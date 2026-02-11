from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests


@dataclass(slots=True)
class EmbeddingClient:
    """외부 임베딩 API(또는 mock)를 호출하는 클라이언트."""

    api_url: str = "mock"
    api_key: str | None = None
    model: str | None = None
    timeout: float = 10.0
    provider: str = "generic"
    mock_dimension: int = 64

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "EmbeddingClient":
        return cls(
            api_url=str(cfg.get("api_url", "mock")),
            api_key=cfg.get("api_key"),
            model=cfg.get("model"),
            timeout=float(cfg.get("timeout", 10.0)),
            provider=str(cfg.get("provider", "generic")),
            mock_dimension=int(cfg.get("mock_dimension", 64)),
        )

    def embed_text(self, text: str) -> np.ndarray:
        if self.provider == "mock" or self.api_url == "mock":
            return np.asarray(self._mock_embedding(text), dtype=np.float32)

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
        return np.asarray([self.embed_text(t) for t in texts], dtype=np.float32)

    def _mock_embedding(self, text: str) -> list[float]:
        digest = sha256(text.encode("utf-8")).digest()
        return [((digest[i % len(digest)] / 255.0) * 2 - 1) for i in range(self.mock_dimension)]


class search_store:
    """content 저장/로드, Mongo CRUD, 임베딩 인덱스 관리를 담당."""

    def __init__(self, embedding_client: EmbeddingClient, id_key: str = "contentId") -> None:
        self.embedding_client = embedding_client
        self.id_key = id_key

        self.contents: list[dict[str, Any]] = []
        self.concat_texts: list[str] = []
        self.concat_embeddings: np.ndarray | None = None
        self.concat_embeddings_norm: np.ndarray | None = None
        self.field_embeddings: dict[str, np.ndarray] = {}
        self.field_embeddings_norm: dict[str, np.ndarray] = {}
        self.content_ids: list[str] = []

    # ---------- content load/save ----------
    def set_contents(self, items: list[dict[str, Any]]) -> None:
        if any(not isinstance(x, dict) for x in items):
            raise ValueError("contents는 List[dict] 형태여야 합니다.")
        self.contents = items
        self.content_ids = [str(c[self.id_key]) for c in self.contents if self.id_key in c]

    def load_contents_from_json(self, json_path: str | Path) -> None:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON 루트는 List[dict] 형태여야 합니다.")
        self.set_contents(data)

    def load_contents_from_mongodb(self, collection: Any, query: dict[str, Any] | None = None, limit: int | None = None) -> None:
        cursor = collection.find(query or {})
        if limit:
            cursor = cursor.limit(limit)
        rows = []
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
            for c in self.contents:
                if self.id_key not in c:
                    raise KeyError(f"'{self.id_key}'가 없는 content는 저장할 수 없습니다.")
                ops.append(UpdateOne({self.id_key: c[self.id_key]}, {"$set": c}, upsert=True))
            if ops:
                collection.bulk_write(ops, ordered=False)
            return
        collection.insert_many(self.contents)

    # ---------- Mongo CRUD by id ----------
    def create_content(self, collection: Any, content: dict[str, Any], upsert: bool = False) -> None:
        if self.id_key not in content:
            raise KeyError(f"'{self.id_key}'가 필요합니다.")
        if upsert:
            collection.update_one({self.id_key: content[self.id_key]}, {"$set": content}, upsert=True)
        else:
            collection.insert_one(content)

    def read_content(self, collection: Any, content_id: str) -> dict[str, Any] | None:
        row = collection.find_one({self.id_key: content_id})
        if not row:
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

        if mode in {"concat", "both"}:
            keys = list(concat_keys) if concat_keys else sorted(self._all_keys())
            self.concat_texts = [self._concat_text(c, keys) for c in self.contents]
            self.concat_embeddings = self.embedding_client.embed_texts(self.concat_texts)
            self.concat_embeddings_norm = self._row_norm(self.concat_embeddings)

        if mode in {"per_key", "both"}:
            fields = list(per_key_fields) if per_key_fields else sorted(self._all_keys())
            self.field_embeddings.clear()
            self.field_embeddings_norm.clear()
            for field in fields:
                texts = [self._to_text(c.get(field, "")) for c in self.contents]
                matrix = self.embedding_client.embed_texts(texts)
                self.field_embeddings[field] = matrix
                self.field_embeddings_norm[field] = self._row_norm(matrix)

        self.content_ids = [str(c[self.id_key]) for c in self.contents if self.id_key in c]

    def save_to_mongodb_split(self, metadata_collection: Any, embedding_collection: Any, upsert: bool = True) -> None:
        if not self.contents:
            return

        metadata_docs = [dict(c) for c in self.contents]
        embedding_docs: list[dict[str, Any]] = []
        n = len(self.contents)

        for i, c in enumerate(self.contents):
            if self.id_key not in c:
                raise KeyError(f"'{self.id_key}'가 없는 content는 split 저장할 수 없습니다.")
            doc: dict[str, Any] = {self.id_key: c[self.id_key]}
            if self.concat_embeddings is not None and len(self.concat_embeddings) == n:
                doc["concat_embedding"] = self.concat_embeddings[i].astype(np.float32).tolist()
            if self.field_embeddings:
                per_field: dict[str, list[float]] = {}
                for field, matrix in self.field_embeddings.items():
                    if len(matrix) == n:
                        per_field[field] = matrix[i].astype(np.float32).tolist()
                if per_field:
                    doc["field_embeddings"] = per_field
            embedding_docs.append(doc)

        self._write_docs_by_id(metadata_collection, metadata_docs, upsert)
        self._write_docs_by_id(embedding_collection, embedding_docs, upsert)

    def load_from_mongodb_split(
        self,
        metadata_collection: Any,
        embedding_collection: Any,
        query: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> None:
        self.load_contents_from_mongodb(metadata_collection, query=query, limit=limit)
        if not self.contents:
            return

        ids = [c[self.id_key] for c in self.contents if self.id_key in c]
        embed_docs = list(embedding_collection.find({self.id_key: {"$in": ids}}))
        embed_by_id = {doc[self.id_key]: doc for doc in embed_docs if self.id_key in doc}

        concat_rows: list[list[float]] = []
        concat_ok = True
        field_rows: dict[str, list[list[float]]] = {}

        for c in self.contents:
            cid = c.get(self.id_key)
            doc = embed_by_id.get(cid)
            if not doc:
                concat_ok = False
                continue

            vec = doc.get("concat_embedding")
            if isinstance(vec, list):
                concat_rows.append(vec)
            else:
                concat_ok = False

            fields = doc.get("field_embeddings", {})
            if isinstance(fields, dict):
                for field, fvec in fields.items():
                    if isinstance(fvec, list):
                        field_rows.setdefault(field, []).append(fvec)

        if concat_ok and len(concat_rows) == len(self.contents):
            self.concat_embeddings = np.asarray(concat_rows, dtype=np.float32)
            self.concat_embeddings_norm = self._row_norm(self.concat_embeddings)

        self.field_embeddings.clear()
        self.field_embeddings_norm.clear()
        for field, rows in field_rows.items():
            if len(rows) != len(self.contents):
                continue
            matrix = np.asarray(rows, dtype=np.float32)
            self.field_embeddings[field] = matrix
            self.field_embeddings_norm[field] = self._row_norm(matrix)

        self.content_ids = [str(c[self.id_key]) for c in self.contents if self.id_key in c]

    # ---------- internal ----------
    def _all_keys(self) -> set[str]:
        keys: set[str] = set()
        for c in self.contents:
            keys.update(c.keys())
        return keys

    @staticmethod
    def _to_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _concat_text(self, content: dict[str, Any], keys: Iterable[str]) -> str:
        return " ".join(self._to_text(content.get(k, "")) for k in keys).strip()

    @staticmethod
    def _row_norm(matrix: np.ndarray) -> np.ndarray:
        return np.linalg.norm(matrix, axis=1) + 1e-12

    def _write_docs_by_id(self, collection: Any, docs: list[dict[str, Any]], upsert: bool) -> None:
        if not docs:
            return
        if upsert:
            from pymongo import UpdateOne

            ops = [UpdateOne({self.id_key: d[self.id_key]}, {"$set": d}, upsert=True) for d in docs]
            collection.bulk_write(ops, ordered=False)
            return
        collection.insert_many(docs)


class search_engine:
    """search_store의 인덱스를 사용해 코사인 유사도 검색을 수행."""

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

    def cosine_similarity(self, query_vector: np.ndarray, matrix: np.ndarray, matrix_norm: np.ndarray) -> np.ndarray:
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
