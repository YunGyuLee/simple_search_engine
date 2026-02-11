# Simple Search Engine (Python 3.11)

요구사항에 맞춰 **두 개의 클래스**로 최소 기능만 구현한 검색 엔진입니다.

- `search_store`: content 로드/저장 + 임베딩 인덱스 구축
- `search_engine`: 코사인 유사도 기반 검색 수행

## 1) 설치

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
store.build_embeddings(mode="both")

engine_concat = search_engine("concat", store, embedding_model)
result = engine_concat.search(keyword="매출 개선", top_k=5)

engine_per_key = search_engine("per_key", store, embedding_model)
result2 = engine_per_key.search(
    keyword_by_field={"contentNm": "매출", "contentDesc": "고객 유지", "contentMetric": "전환율"},
    weight_by_field={"contentNm": 0.5, "contentDesc": 0.3, "contentMetric": 0.2},
    top_k=5,
)
```

## 4) 임베딩 API 연결

- URL: `embedding_model["api_url"]`
- Header: `Authorization: Bearer <api_key>` (api_key 존재 시)
- Body: `{ "input": "...", "model": "..." }`

응답은 아래 중 하나를 지원합니다.
1. `{ "embedding": [ ... ] }`
2. `{ "data": [{ "embedding": [ ... ] }] }`

개발/테스트용으로 `provider=mock` 또는 `api_url=mock`이면 해시 기반 deterministic 임베딩을 사용합니다.

## 5) MongoDB 단일 collection 사용

```python
from pymongo import MongoClient
from simple_search import search_store

client = MongoClient("mongodb://localhost:27017")
store = search_store(embedding_model={"provider": "mock", "api_url": "mock"})

store.load_from_mongodb(client, "search_db", "contents")
store.save_contents_to_mongodb(client, "search_db", "contents", replace_by_content_id=True)
```

## 6) MongoDB 분리 저장/로드 (필수 메타 + 임베딩)

`contentId` 기준으로 아래 두 collection으로 분리할 수 있습니다.

- 메타 collection: 필수 메타정보(`REQUIRED_CONTENT_KEYS`)
- 임베딩 collection: `concat_embedding`, `field_embeddings`

```python
from pymongo import MongoClient
from simple_search import search_store

client = MongoClient("mongodb://localhost:27017")
store = search_store(embedding_model={"provider": "mock", "api_url": "mock"})

store.load_from_json("sample_contents.json")
store.build_embeddings(mode="both")

# 저장 (분리)
store.save_to_mongodb_split(
    mongo_client=client,
    db_name="search_db",
    metadata_collection_name="content_metadata",
    embedding_collection_name="content_embeddings",
    replace_by_content_id=True,
)

# 로드 (분리)
store.load_from_mongodb_split(
    mongo_client=client,
    db_name="search_db",
    metadata_collection_name="content_metadata",
    embedding_collection_name="content_embeddings",
)
```

### 6-1) 실제 MongoDB 문서 저장 예시

아래는 `contentId = "C001"`인 데이터가 저장되는 예시입니다.

#### metadata collection (`content_metadata`)

```json
{
  "contentId": "C001",
  "contentNm": "매출 개선 전략",
  "contentDesc": "전환율 향상을 위한 마케팅 캠페인",
  "contentGC": "A",
  "contentCusGC": "B",
  "contentPeriod": "2025Q1",
  "contentObject": "매출",
  "contentAction": "캠페인",
  "contentMetric": "전환율"
}
```

#### embedding collection (`content_embeddings`)

embedding mode(`build_embeddings(mode=...)`)에 따라 문서 형태가 달라집니다.

1) `mode="concat"`

```json
{
  "contentId": "C001",
  "concat_embedding": [0.12, -0.03, 0.55, -0.21]
}
```

2) `mode="per_key"`

```json
{
  "contentId": "C001",
  "field_embeddings": {
    "contentNm": [0.01, 0.12, -0.05, 0.89],
    "contentDesc": [0.45, -0.11, 0.03, 0.22],
    "contentMetric": [0.31, 0.09, -0.28, 0.10]
  }
}
```

3) `mode="both"`

```json
{
  "contentId": "C001",
  "concat_embedding": [0.12, -0.03, 0.55, -0.21],
  "field_embeddings": {
    "contentNm": [0.01, 0.12, -0.05, 0.89],
    "contentDesc": [0.45, -0.11, 0.03, 0.22],
    "contentMetric": [0.31, 0.09, -0.28, 0.10]
  }
}
```

> 참고: 실제 벡터 길이는 사용 중인 임베딩 모델 차원(`mock_dimension` 또는 API 모델 차원)과 동일합니다.

## 7) 디버깅 순서 추천

1. `load_from_json` 또는 `load_from_mongodb`/`load_from_mongodb_split`
2. `build_embeddings(mode="concat" | "per_key" | "both")`
3. `search_engine(...).search(...)`
