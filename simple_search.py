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
    """Embedding 호출 설정 DTO.

    in:
      - api_url: 임베딩 API URL 또는 'mock'
      - api_key: 인증 키 (없으면 None)
      - model: 모델명 (없으면 None)
      - timeout: HTTP timeout (초)
      - provider: provider 식별자
      - mock_dimension: mock 임베딩 벡터 차원
    out:
      - dataclass 인스턴스
    desc:
      - embedding_model(dict)를 안정적으로 파싱해 내부 처리에 사용한다.
    """

    api_url: str
    api_key: str | None
    model: str | None
    timeout: float
    provider: str
    mock_dimension: int


class search_store:
    """검색 대상 저장소 + 임베딩 인덱스 관리 클래스.

    in:
      - embedding_model: 임베딩 API 설정(dict)
    out:
      - search_store 인스턴스
    desc:
      - content 원본 데이터 로드/검증/저장, 임베딩 생성 및 인덱스 캐시를 담당한다.
    """

    def __init__(self, embedding_model: dict[str, Any]) -> None:
        """초기화.

        in:
          - embedding_model: api_url, api_key, model 등을 담은 dict
        out:
          - None
        desc:
          - content 및 임베딩 캐시용 멤버 변수를 초기 상태로 준비한다.
        """
        self.contents: list[dict[str, Any]] = []
        self.embedding_model: dict[str, Any] = embedding_model

        self.concat_texts: list[str] = []
        self.concat_embeddings: np.ndarray | None = None
        self.concat_embeddings_norm: np.ndarray | None = None

        self.field_embeddings: dict[str, np.ndarray] = {}
        self.field_embeddings_norm: dict[str, np.ndarray] = {}
        self._content_ids: list[str] = []

    def load_from_json(self, json_path: str | Path) -> None:
        """JSON 파일에서 contents를 로드한다.

        in:
          - json_path: List[dict] 구조 JSON 파일 경로
        out:
          - None
        desc:
          - 파일을 읽고 필수 key 검증 후 self.contents에 반영한다.
        """
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
        """MongoDB에서 contents를 로드한다(외부 client 주입).

        in:
          - mongo_client: 외부에서 생성한 MongoClient
          - db_name: DB 이름
          - collection_name: 컬렉션 이름
          - query: 조회 조건 (기본 {})
          - limit: 최대 조회 개수 (기본 None)
        out:
          - None
        desc:
          - 조회 결과의 _id를 제거하고 필수 key 검증 후 self.contents에 반영한다.
        """
        query = query or {}
        collection = mongo_client[db_name][collection_name]

        cursor = collection.find(query)
        if limit:
            cursor = cursor.limit(limit)

        items = [self._normalize_mongo_row(row) for row in cursor]
        self._set_contents(items)

    def save_contents_to_mongodb(
        self,
        mongo_client: Any,
        db_name: str,
        collection_name: str,
        replace_by_content_id: bool = True,
    ) -> None:
        """현재 contents를 MongoDB에 저장한다(외부 client 주입).

        in:
          - mongo_client: 외부에서 생성한 MongoClient
          - db_name: DB 이름
          - collection_name: 컬렉션 이름
          - replace_by_content_id: True면 upsert, False면 insert_many
        out:
          - None
        desc:
          - contentId 기준 upsert 또는 단순 insert 방식으로 저장한다.
        """
        if not self.contents:
            return

        collection = mongo_client[db_name][collection_name]

        if replace_by_content_id:
            ops = [self._build_upsert_operation(content) for content in self.contents]
            collection.bulk_write(ops, ordered=False)
            return

        collection.insert_many(self.contents)

    def build_embeddings(
        self,
        mode: str = "both",
        concat_keys: Iterable[str] | None = None,
        per_key_fields: Iterable[str] | None = None,
    ) -> None:
        """임베딩 인덱스를 생성한다.

        in:
          - mode: 'concat' | 'per_key' | 'both'
          - concat_keys: concat 임베딩에 사용할 key 목록
          - per_key_fields: field별 임베딩 대상 key 목록
        out:
          - None
        desc:
          - mode에 맞는 임베딩 matrix와 norm 캐시를 생성해 검색 속도를 높인다.
        """
        self._validate_before_embedding(mode)
        concat_keys = list(concat_keys or REQUIRED_CONTENT_KEYS)
        per_key_fields = list(per_key_fields or REQUIRED_CONTENT_KEYS)

        if mode in ("concat", "both"):
            self._build_concat_embeddings(concat_keys)

        if mode in ("per_key", "both"):
            self._build_per_key_embeddings(per_key_fields)

        self._content_ids = [str(c["contentId"]) for c in self.contents]

    def _set_contents(self, items: list[dict[str, Any]]) -> None:
        """contents를 검증 후 내부 상태에 설정한다.

        in:
          - items: content dict 리스트
        out:
          - None
        desc:
          - 각 row가 dict인지와 필수 key 존재 여부를 검사한다.
        """
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
        """Mongo 문서에서 내부 사용 형태로 정규화한다.

        in:
          - row: MongoDB 문서(dict)
        out:
          - _id 제거된 dict
        desc:
          - 내부 content 스키마 검증 전에 Mongo 고유 필드를 제거한다.
        """
        row = dict(row)
        row.pop("_id", None)
        return row

    @staticmethod
    def _build_upsert_operation(content: dict[str, Any]) -> Any:
        """content 1건에 대한 Mongo upsert 연산 객체를 만든다.

        in:
          - content: 저장 대상 content dict
        out:
          - pymongo UpdateOne 연산 객체
        desc:
          - contentId 기준으로 upsert 동작을 구성한다.
        """
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
        """임베딩 생성 전 필수 조건을 검증한다.

        in:
          - mode: 임베딩 모드 문자열
        out:
          - None
        desc:
          - 지원 모드인지, contents가 비어있지 않은지 확인한다.
        """
        if mode not in ("concat", "per_key", "both"):
            raise ValueError("mode는 'concat', 'per_key', 'both' 중 하나여야 합니다.")
        if not self.contents:
            raise ValueError("contents가 비어 있습니다. 먼저 DB/JSON을 로드하세요.")

    def _build_concat_embeddings(self, concat_keys: list[str]) -> None:
        """concat 텍스트 임베딩 matrix를 생성한다.

        in:
          - concat_keys: concat 대상 key 목록
        out:
          - None
        desc:
          - 각 content의 지정 key를 합쳐 임베딩하고 norm 캐시를 계산한다.
        """
        self.concat_texts = [self._concat_content_text(c, concat_keys) for c in self.contents]
        self.concat_embeddings = self._embed_texts(self.concat_texts)
        self.concat_embeddings_norm = self._row_norm(self.concat_embeddings)

    def _build_per_key_embeddings(self, per_key_fields: list[str]) -> None:
        """field별 임베딩 matrix를 생성한다.

        in:
          - per_key_fields: field 임베딩 대상 key 목록
        out:
          - None
        desc:
          - 각 field마다 content 전부를 임베딩해 field별 matrix/norm 캐시를 구성한다.
        """
        self.field_embeddings.clear()
        self.field_embeddings_norm.clear()

        for field in per_key_fields:
            texts = [self._to_text(c.get(field, "")) for c in self.contents]
            matrix = self._embed_texts(texts)
            self.field_embeddings[field] = matrix
            self.field_embeddings_norm[field] = self._row_norm(matrix)

    @staticmethod
    def _to_text(value: Any) -> str:
        """임의 값을 임베딩 가능한 문자열로 변환한다.

        in:
          - value: 원본 값(문자열/숫자/객체/None)
        out:
          - 문자열
        desc:
          - None은 빈 문자열, str은 그대로, 그 외는 JSON 문자열로 직렬화한다.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _concat_content_text(self, content: dict[str, Any], keys: Iterable[str]) -> str:
        """content의 여러 key 값을 하나의 문자열로 합친다.

        in:
          - content: content dict
          - keys: concat 대상 key 시퀀스
        out:
          - 공백 결합 문자열
        desc:
          - 검색용 통합 텍스트를 만들 때 사용한다.
        """
        return " ".join(self._to_text(content.get(key, "")) for key in keys).strip()

    @staticmethod
    def _row_norm(matrix: np.ndarray) -> np.ndarray:
        """행 단위 L2 norm 벡터를 계산한다.

        in:
          - matrix: shape=(n, d) 임베딩 행렬
        out:
          - shape=(n,) norm 배열
        desc:
          - 0 division 방지를 위해 작은 epsilon을 더한다.
        """
        return np.linalg.norm(matrix, axis=1) + 1e-12

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """문자열 리스트를 임베딩 행렬로 변환한다.

        in:
          - texts: 임베딩할 텍스트 리스트
        out:
          - float32 임베딩 행렬
        desc:
          - 내부 단건 임베딩 함수를 반복 호출해 batch 형태로 묶는다.
        """
        vectors = [self._embed_single_text(t) for t in texts]
        return np.asarray(vectors, dtype=np.float32)

    def _embed_single_text(self, text: str) -> list[float]:
        """텍스트 1건을 임베딩한다.

        in:
          - text: 임베딩할 문자열
        out:
          - 임베딩 벡터(list[float])
        desc:
          - mock 모드면 로컬 생성, 아니면 외부 API 호출 후 응답에서 벡터를 추출한다.
        """
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
        """해시 기반 deterministic mock 임베딩을 생성한다.

        in:
          - text: 입력 텍스트
          - dim: 벡터 차원
        out:
          - 길이 dim 벡터(list[float])
        desc:
          - 외부 API 없이 재현 가능한 테스트 임베딩을 제공한다.
        """
        digest = sha256(text.encode("utf-8")).digest()
        result = []
        for i in range(dim):
            b = digest[i % len(digest)]
            result.append((b / 255.0) * 2 - 1)
        return result

    @staticmethod
    def _parse_embedding_config(model_cfg: dict[str, Any]) -> _EmbeddingConfig:
        """embedding_model dict를 내부 설정 객체로 변환한다.

        in:
          - model_cfg: 사용자 설정 dict
        out:
          - _EmbeddingConfig
        desc:
          - 누락된 항목은 기본값으로 채워 안정적인 호출을 보장한다.
        """
        return _EmbeddingConfig(
            api_url=str(model_cfg.get("api_url", "mock")),
            api_key=model_cfg.get("api_key"),
            model=model_cfg.get("model"),
            timeout=float(model_cfg.get("timeout", 10.0)),
            provider=str(model_cfg.get("provider", "generic")),
            mock_dimension=int(model_cfg.get("mock_dimension", 64)),
        )


class search_engine:
    """search_store 기반 코사인 유사도 검색 엔진.

    in:
      - embedding_mode: 'concat' 또는 'per_key'
      - store: search_store 인스턴스
      - embedding_model: 임베딩 설정(dict)
    out:
      - search_engine 인스턴스
    desc:
      - query 임베딩 + 벡터 연산으로 빠르게 유사도를 계산하고 top-k를 반환한다.
    """

    def __init__(
        self,
        embedding_mode: str,
        store: search_store,
        embedding_model: dict[str, Any],
    ) -> None:
        """초기화.

        in:
          - embedding_mode: 'concat' | 'per_key'
          - store: 검색 대상 저장소
          - embedding_model: 임베딩 모델 설정
        out:
          - None
        desc:
          - 모드 검증 후 엔진이 참조할 상태를 저장한다.
        """
        if embedding_mode not in {"concat", "per_key"}:
            raise ValueError("embedding_mode는 'concat' 또는 'per_key'여야 합니다.")
        self.embedding_mode = embedding_mode
        self.search_store = store
        self.embedding_model = embedding_model

    @classmethod
    def from_config(cls, config_path: str | Path, store: search_store) -> "search_engine":
        """설정 파일로 엔진을 초기화한다.

        in:
          - config_path: JSON 설정 파일 경로
          - store: search_store 인스턴스
        out:
          - search_engine 인스턴스
        desc:
          - embedding_mode/embedding_model을 읽어 엔진 생성에 전달한다.
        """
        path = Path(config_path)
        config = json.loads(path.read_text(encoding="utf-8"))
        mode = config.get("embedding_mode", "concat")
        embedding_model = config.get("embedding_model", {})
        return cls(embedding_mode=mode, store=store, embedding_model=embedding_model)

    def cosine_similarity(self, query_vector: np.ndarray, matrix: np.ndarray, matrix_norm: np.ndarray) -> np.ndarray:
        """코사인 유사도를 벡터 연산으로 계산한다.

        in:
          - query_vector: shape=(d,) 쿼리 벡터
          - matrix: shape=(n, d) 문서 임베딩 행렬
          - matrix_norm: shape=(n,) 문서 행 norm
        out:
          - shape=(n,) 유사도 점수
        desc:
          - dot product와 norm 캐시를 활용해 빠르게 유사도를 계산한다.
        """
        query_norm = float(np.linalg.norm(query_vector) + 1e-12)
        dots = matrix @ query_vector
        return dots / (matrix_norm * query_norm)

    def search_concat(self, keyword: str, top_k: int = 10) -> list[dict[str, Any]]:
        """단일 키워드 vs concat 임베딩 검색.

        in:
          - keyword: 검색어
          - top_k: 반환 개수
        out:
          - score 포함 content dict 리스트
        desc:
          - keyword를 임베딩해 concat 임베딩 인덱스와 코사인 유사도를 계산한다.
        """
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
        """field별 키워드 + 가중치 기반 검색.

        in:
          - keyword_by_field: {field: keyword}
          - weight_by_field: {field: weight} (기본 1.0)
          - top_k: 반환 개수
        out:
          - score 포함 content dict 리스트
        desc:
          - 각 field 유사도에 가중치를 반영해 최종 점수를 산출한다.
        """
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
        """엔진 모드에 따라 검색을 분기 실행한다.

        in:
          - keyword: concat 모드 검색어
          - keyword_by_field: per_key 모드 검색어 dict
          - weight_by_field: per_key 모드 가중치 dict
          - top_k: 반환 개수
        out:
          - score 포함 content dict 리스트
        desc:
          - concat/per_key 모드에 맞는 검색 함수를 호출한다.
        """
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
        """concat 검색 가능 상태를 검증한다.

        in:
          - 없음
        out:
          - None
        desc:
          - concat 임베딩 캐시 존재 여부를 확인한다.
        """
        if self.search_store.concat_embeddings is None or self.search_store.concat_embeddings_norm is None:
            raise ValueError("concat 임베딩이 없습니다. search_store.build_embeddings(mode='concat' or 'both')를 먼저 호출하세요.")

    def _validate_per_key_ready(self) -> None:
        """per_key 검색 가능 상태를 검증한다.

        in:
          - 없음
        out:
          - None
        desc:
          - field 임베딩 캐시 존재 여부를 확인한다.
        """
        if not self.search_store.field_embeddings:
            raise ValueError("per_key 임베딩이 없습니다. search_store.build_embeddings(mode='per_key' or 'both')를 먼저 호출하세요.")

    def _top_k(self, scores: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        """유사도 배열에서 상위 k개 결과를 반환한다.

        in:
          - scores: shape=(n,) 점수 배열
          - top_k: 반환 개수
        out:
          - score 포함 content dict 리스트
        desc:
          - argpartition + 부분 정렬로 top-k를 효율적으로 추출한다.
        """
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
