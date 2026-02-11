# Simple Search Engine (Python 3.11)

요구사항에 맞춰 **두 개의 클래스**로 최소 기능만 구현한 검색 엔진입니다.

- `search_store`: content 로드/저장 + 임베딩 인덱스 구축
- `search_engine`: 코사인 유사도 기반 검색 수행

## 1) 설치

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install numpy requests
# MongoDB 기능이 필요하면
pip install pymongo
```

## 2) content 형식

`contents`는 `List[dict]` 형태이며 각 dict는 아래 key를 필수로 가져야 합니다.

- `contentId`
- `contentNm`
- `contentDesc`
- `contentGC`
- `contentCusGC`
- `contentPeriod`
- `contentObject`
- `contentAction`
- `contentMetric`

## 3) 기본 사용법

```python
from simple_search import search_store, search_engine

embedding_model = {
    "provider": "mock",
    "api_url": "mock",
    "api_key": "",
    "model": "",
    "mock_dimension": 64,
}

store = search_store(embedding_model=embedding_model)
store.load_from_json("sample_contents.json")

# concat 임베딩 + per_key 임베딩 모두 생성
store.build_embeddings(mode="both")

# concat 검색
engine_concat = search_engine("concat", store, embedding_model)
result = engine_concat.search(keyword="매출 개선", top_k=5)

# per_key 가중 검색
engine_per_key = search_engine("per_key", store, embedding_model)
result2 = engine_per_key.search(
    keyword_by_field={
        "contentNm": "매출",
        "contentDesc": "고객 유지",
        "contentMetric": "전환율"
    },
    weight_by_field={
        "contentNm": 0.5,
        "contentDesc": 0.3,
        "contentMetric": 0.2,
    },
    top_k=5,
)
```

## 4) 임베딩 API 연결

`search_store` 내부에서 임베딩 API를 다음 형태로 호출합니다.

- URL: `embedding_model["api_url"]`
- Header: `Authorization: Bearer <api_key>` (api_key 존재 시)
- Body: `{ "input": "...", "model": "..." }`

응답은 아래 중 하나를 지원합니다.

1. `{ "embedding": [ ... ] }`
2. `{ "data": [{ "embedding": [ ... ] }] }`

개발/테스트를 위해 `provider=mock` 또는 `api_url=mock`이면 해시 기반 deterministic 임베딩을 사용합니다.

## 5) MongoDB 사용

```python
store.load_from_mongodb(
    mongo_uri="mongodb://localhost:27017",
    db_name="search_db",
    collection_name="contents",
)

store.save_contents_to_mongodb(
    mongo_uri="mongodb://localhost:27017",
    db_name="search_db",
    collection_name="contents",
    replace_by_content_id=True,
)
```
