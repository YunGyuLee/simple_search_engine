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
        """
        input:
          - cfg(dict): embedding_model 설정 dict (api_url, api_key, model, timeout)
        output:
          - EmbeddingClient 인스턴스
        desc:
          - 설정 dict를 검증/파싱하여 EmbeddingClient를 생성한다.
          - api_url이 없으면 예외를 발생시킨다.
        """
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
        """
        input:
          - text(str): 임베딩할 텍스트
        output:
          - np.ndarray(shape=(d,), dtype=float32): 임베딩 벡터
        desc:
          - 단일 텍스트를 외부 API에 전달하고 응답에서 임베딩 벡터를 추출한다.
          - 지원 응답 포맷:
            1) {"embedding": [...]}
            2) {"data": [{"embedding": [...]}]}
        """
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
        """
        input:
          - texts(list[str]): 임베딩할 텍스트 리스트
        output:
          - np.ndarray(shape=(n, d), dtype=float32): 임베딩 행렬
        desc:
          - 텍스트 리스트를 순차적으로 임베딩하여 2차원 행렬로 반환한다.
        """
        vectors = [self.embed_text(t) for t in texts]
        return np.asarray(vectors, dtype=np.float32)


class search_store:
    """ID 기준 Mongo CRUD + 벌크 load/save + 임베딩 인덱스 관리."""

    def __init__(self, embedding_client: EmbeddingClient, id_key: str = "contentId") -> None:
        """
        input:
          - embedding_client(EmbeddingClient): 임베딩 API 클라이언트
          - id_key(str): 문서 식별자 key (기본 'contentId')
        output:
          - None
        desc:
          - content 캐시, 임베딩 캐시(concat/per_key), id 캐시를 초기화한다.
        """
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
    def replace_contents(self, items: list[dict[str, Any]]) -> None:
        """
        input:
          - items(list[dict]): store에 반영할 content 목록
        output:
          - None
        desc:
          - 기존 self.contents를 입력 items로 교체한다.
          - id_key가 있는 항목에 대해 self.content_ids를 동기화한다.
        """
        if any(not isinstance(item, dict) for item in items):
            raise ValueError("contents는 List[dict] 형태여야 합니다.")
        self.contents = items
        self.content_ids = [str(item[self.id_key]) for item in items if self.id_key in item]

    def load_contents_from_json(self, json_path: str | Path) -> None:
        """
        input:
          - json_path(str|Path): JSON 파일 경로
        output:
          - None
        desc:
          - JSON 파일(List[dict])을 읽어 self.contents에 반영한다.
        """
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
        """
        input:
          - collection: pymongo collection 객체
          - query(dict|None): 조회 조건
          - limit(int|None): 조회 최대 개수
          - load_embeddings_if_present(bool): True면 문서 내 저장된 임베딩을 인메모리 캐시에 로드 시도
          - embedding_field(str): 문서 내 임베딩 저장 필드명 (기본 'embeddings')
        output:
          - None
        desc:
          - MongoDB에서 문서를 벌크 로드하여 self.contents에 반영한다.
          - load_embeddings_if_present=True이면, 로드된 문서의 embedding_field를 파싱해
            concat/per_key 임베딩 캐시를 재구성한다.
        """
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
        """
        input:
          - collection: pymongo collection 객체
          - upsert(bool): True면 id_key 기준 upsert, False면 insert_many
        output:
          - None
        desc:
          - 현재 self.contents를 MongoDB에 저장한다.
          - upsert=True일 때는 id_key가 없는 문서 저장을 허용하지 않는다.
        """
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
        """
        input:
          - collection: pymongo collection 객체
          - content(dict): 저장할 단건 문서
          - upsert(bool): True면 id_key 기준 upsert
        output:
          - None
        desc:
          - id_key 기반 단건 생성(또는 upsert) 작업을 수행한다.
        """
        if self.id_key not in content:
            raise KeyError(f"'{self.id_key}'가 필요합니다.")
        if upsert:
            collection.update_one({self.id_key: content[self.id_key]}, {"$set": content}, upsert=True)
            return
        collection.insert_one(content)

    def read_content(self, collection: Any, content_id: str) -> dict[str, Any] | None:
        """
        input:
          - collection: pymongo collection 객체
          - content_id(str): 조회할 문서 id
        output:
          - dict | None: 조회 문서(_id 제외) 또는 None
        desc:
          - id_key 기준 단건 조회를 수행한다.
        """
        row = collection.find_one({self.id_key: content_id})
        if row is None:
            return None
        row = dict(row)
        row.pop("_id", None)
        return row

    def update_content(self, collection: Any, content_id: str, patch: dict[str, Any]) -> bool:
        """
        input:
          - collection: pymongo collection 객체
          - content_id(str): 수정할 문서 id
          - patch(dict): $set에 적용할 변경 필드
        output:
          - bool: 문서가 존재해 업데이트 되었으면 True
        desc:
          - id_key 기준 단건 부분 업데이트를 수행한다.
        """
        result = collection.update_one({self.id_key: content_id}, {"$set": patch})
        return result.matched_count > 0

    def delete_content(self, collection: Any, content_id: str) -> bool:
        """
        input:
          - collection: pymongo collection 객체
          - content_id(str): 삭제할 문서 id
        output:
          - bool: 삭제 성공 시 True
        desc:
          - id_key 기준 단건 삭제를 수행한다.
        """
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
        """
        input:
          - mode(str): 'concat' | 'per_key' | 'both'
          - concat_keys(Iterable[str]|None): concat 텍스트 생성 key 목록
          - per_key_fields(Iterable[str]|None): per_key 임베딩 key 목록
          - reuse_cached_embeddings(bool): True면 문서 내 저장 임베딩을 우선 로드해 API 호출 생략 시도
          - embedding_field(str): 문서 내 임베딩 저장 필드명
        output:
          - None
        desc:
          - 검색용 인메모리 임베딩 캐시(concat/per_key)를 생성한다.
          - reuse_cached_embeddings=True이고 cached 임베딩이 완전하면 API 호출 없이 로드 결과를 사용한다.
          - 필요한 캐시가 불완전할 경우에만 API 호출로 재생성한다.
        """
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
        """
        input:
          - collection: pymongo collection 객체
          - content_id(str): 임베딩할 대상 문서 id
          - concat_keys(Iterable[str]|None): concat 임베딩에 사용할 key 목록
          - per_key_fields(Iterable[str]|None): per_key 임베딩에 사용할 key 목록
          - embedding_field(str): 임베딩 저장 필드명
        output:
          - dict: 저장된 임베딩 payload {concat, per_key}
        desc:
          - id 기준으로 문서를 조회해 단건 임베딩을 생성하고,
            동일 문서의 embedding_field에 결과를 저장한다.
        """
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
        skip_if_embedding_exists: bool = True,
    ) -> int:
        """
        input:
          - collection: pymongo collection 객체
          - query(dict|None): 대상 문서 조회 조건
          - limit(int|None): 대상 최대 개수
          - concat_keys/per_key_fields: 임베딩 key 목록
          - embedding_field(str): 임베딩 저장 필드명
          - skip_if_embedding_exists(bool): True면 embedding_field가 이미 존재하는 문서는 건너뜀
        output:
          - int: 실제 업데이트 건수
        desc:
          - 조회된 문서 집합을 bulk 임베딩 후 MongoDB에 bulk update한다.
          - skip_if_embedding_exists=True면 기존 임베딩이 있는 문서는 재임베딩하지 않는다.
        """
        self.load_contents_from_mongodb(
            collection,
            query=query,
            limit=limit,
            load_embeddings_if_present=False,
            embedding_field=embedding_field,
        )
        if not self.contents:
            return 0

        work_items = self.contents
        if skip_if_embedding_exists:
            work_items = [item for item in self.contents if embedding_field not in item]

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

            payload = {"concat": concat_vector, "per_key": per_key_payload}
            ops.append(UpdateOne({self.id_key: item[self.id_key]}, {"$set": {embedding_field: payload}}))

        if ops:
            collection.bulk_write(ops, ordered=False)
        return len(ops)

    def load_cached_embeddings_from_contents(self, embedding_field: str = "embeddings") -> None:
        """
        input:
          - embedding_field(str): 문서 내 임베딩 저장 필드명
        output:
          - None
        desc:
          - self.contents에 이미 저장된 embedding payload를 파싱해
            concat/per_key 인메모리 캐시를 구성한다.
          - 모든 문서에서 일관되게 존재하는 임베딩만 캐시에 반영한다.
        """
        if not self.contents:
            return

        concat_rows: list[list[float]] = []
        concat_ready = True
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

            per_key = embedded.get("per_key", {})
            if isinstance(per_key, dict):
                for field, vec in per_key.items():
                    if isinstance(vec, list):
                        per_key_rows.setdefault(field, []).append(vec)

        if concat_ready and len(concat_rows) == len(self.contents):
            self.concat_embeddings = np.asarray(concat_rows, dtype=np.float32)
            self.concat_embeddings_norm = self._compute_row_l2_norms(self.concat_embeddings)

        self.field_embeddings.clear()
        self.field_embeddings_norm.clear()
        for field, rows in per_key_rows.items():
            if len(rows) != len(self.contents):
                continue
            matrix = np.asarray(rows, dtype=np.float32)
            self.field_embeddings[field] = matrix
            self.field_embeddings_norm[field] = self._compute_row_l2_norms(matrix)

    # ---------- internal ----------
    @staticmethod
    def _collect_content_keys(items: list[dict[str, Any]] | None = None) -> set[str]:
        """
        input:
          - items(list[dict]|None): key를 수집할 대상 문서 목록
        output:
          - set[str]: 전체 key 집합
        desc:
          - 대상 문서들의 key를 합집합으로 수집한다.
        """
        if items is None:
            return set()
        keys: set[str] = set()
        for item in items:
            keys.update(item.keys())
        return keys

    @staticmethod
    def _stringify_value_for_embedding(value: Any) -> str:
        """
        input:
          - value(Any): 원본 값
        output:
          - str: 임베딩용 문자열
        desc:
          - None은 빈 문자열, 문자열은 그대로, 그 외 타입은 JSON 문자열로 변환한다.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _build_concat_text(self, item: dict[str, Any], keys: Iterable[str]) -> str:
        """
        input:
          - item(dict): 단건 content
          - keys(Iterable[str]): 결합할 key 목록
        output:
          - str: concat 텍스트
        desc:
          - 지정 key 값을 순서대로 문자열화 후 공백으로 연결한다.
        """
        return " ".join(self._stringify_value_for_embedding(item.get(k, "")) for k in keys).strip()

    @staticmethod
    def _compute_row_l2_norms(matrix: np.ndarray) -> np.ndarray:
        """
        input:
          - matrix(np.ndarray): shape=(n, d) 행렬
        output:
          - np.ndarray: shape=(n,) L2 norm 배열
        desc:
          - 각 행의 L2 norm을 계산하며 0 division 방지를 위해 epsilon을 더한다.
        """
        return np.linalg.norm(matrix, axis=1) + 1e-12

    def _is_concat_cache_ready(self) -> bool:
        """
        input:
          - 없음
        output:
          - bool: concat 임베딩 캐시 완전성
        desc:
          - concat 임베딩/노름 캐시가 현재 contents 크기와 일치하는지 확인한다.
        """
        return (
            self.concat_embeddings is not None
            and self.concat_embeddings_norm is not None
            and len(self.concat_embeddings) == len(self.contents)
            and len(self.concat_embeddings_norm) == len(self.contents)
        )

    def _is_per_key_cache_ready(self, required_fields: list[str]) -> bool:
        """
        input:
          - required_fields(list[str]): 반드시 있어야 할 per_key field 목록
        output:
          - bool: per_key 캐시 완전성
        desc:
          - required_fields 각각에 대해 임베딩/노름 캐시가 존재하고
            현재 contents 크기와 행 수가 일치하는지 확인한다.
        """
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


class search_engine:
    """search_store 인덱스를 활용한 코사인 유사도 검색 엔진."""

    def __init__(self, embedding_mode: str, store: search_store, embedding_client: EmbeddingClient) -> None:
        """
        input:
          - embedding_mode(str): 'concat' 또는 'per_key'
          - store(search_store): 검색 대상 저장소
          - embedding_client(EmbeddingClient): 쿼리 임베딩 생성 클라이언트
        output:
          - None
        desc:
          - 검색 모드와 의존 객체를 설정한다.
        """
        if embedding_mode not in {"concat", "per_key"}:
            raise ValueError("embedding_mode는 'concat' 또는 'per_key'여야 합니다.")
        self.embedding_mode = embedding_mode
        self.search_store = store
        self.embedding_client = embedding_client

    @classmethod
    def from_config(cls, config_path: str | Path, store: search_store) -> "search_engine":
        """
        input:
          - config_path(str|Path): 설정 파일 경로
          - store(search_store): 검색 대상 저장소
        output:
          - search_engine 인스턴스
        desc:
          - 설정 파일에서 embedding_mode/embedding_model을 읽어 엔진을 생성한다.
        """
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        mode = config.get("embedding_mode", "concat")
        client = EmbeddingClient.from_config(config.get("embedding_model", {}))
        return cls(mode, store, client)

    @staticmethod
    def cosine_similarity(query_vector: np.ndarray, matrix: np.ndarray, matrix_norm: np.ndarray) -> np.ndarray:
        """
        input:
          - query_vector(np.ndarray): shape=(d,) 쿼리 벡터
          - matrix(np.ndarray): shape=(n, d) 문서 임베딩 행렬
          - matrix_norm(np.ndarray): shape=(n,) 문서 임베딩 norm
        output:
          - np.ndarray: shape=(n,) 코사인 유사도 점수
        desc:
          - 쿼리와 문서 임베딩 간 코사인 유사도를 벡터 연산으로 계산한다.
        """
        query_norm = float(np.linalg.norm(query_vector) + 1e-12)
        return (matrix @ query_vector) / (matrix_norm * query_norm)

    def search_concat(self, keyword: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        input:
          - keyword(str): 검색어
          - top_k(int): 반환 개수
        output:
          - list[dict]: score 포함 검색 결과
        desc:
          - keyword를 임베딩해 concat 임베딩 인덱스로 검색한다.
        """
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
        """
        input:
          - keyword_by_field(dict[str,str]): field별 검색어
          - weight_by_field(dict[str,float]|None): field별 가중치
          - top_k(int): 반환 개수
        output:
          - list[dict]: score 포함 검색 결과
        desc:
          - field별 유사도를 계산하고 가중 평균으로 최종 점수를 산출한다.
        """
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
        """
        input:
          - keyword(str|None): concat 모드 검색어
          - keyword_by_field(dict|None): per_key 모드 검색어
          - weight_by_field(dict|None): per_key 모드 가중치
          - top_k(int): 반환 개수
        output:
          - list[dict]: score 포함 검색 결과
        desc:
          - 엔진 모드에 따라 concat/per_key 검색을 분기 실행한다.
        """
        if self.embedding_mode == "concat":
            if not keyword:
                raise ValueError("concat 모드에서는 keyword가 필요합니다.")
            return self.search_concat(keyword=keyword, top_k=top_k)

        if not keyword_by_field:
            raise ValueError("per_key 모드에서는 keyword_by_field가 필요합니다.")
        return self.search_multi_weighted(keyword_by_field, weight_by_field, top_k)

    def _select_top_k_results(self, scores: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        """
        input:
          - scores(np.ndarray): shape=(n,) 점수 배열
          - top_k(int): 반환 개수
        output:
          - list[dict]: score 포함 상위 결과
        desc:
          - argpartition + 부분 정렬로 효율적으로 상위 k개를 선택한다.
        """
        n = len(self.search_store.contents)
        if n == 0:
            return []

        k = max(1, min(top_k, n))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        return [{**self.search_store.contents[i], "score": float(scores[i])} for i in idx.tolist()]
