# Simple Search Engine (Python 3.11)

간단한 검색 엔진을 아래 3개 객체로 분리했습니다.

- `EmbeddingClient`: 임베딩 API(mock 포함)
- `search_store`: content 로드/저장, Mongo CRUD(by id), 임베딩 인덱스 관리
- `search_engine`: 코사인 유사도 검색

> 참고: 컨텐츠 필수 key는 코드에서 강제하지 않습니다. (`id_key`만 CRUD/저장 기준으로 사용)

## 1) 설치

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) 기본 사용법

```python
from simple_search import EmbeddingClient, search_store, search_engine

client = EmbeddingClient(api_url="mock", provider="mock", mock_dimension=64)
store = search_store(embedding_client=client, id_key="contentId")

store.load_contents_from_json("sample_contents.json")
store.build_embeddings(mode="both", concat_keys=["contentNm", "contentDesc"], per_key_fields=["contentNm", "contentDesc", "contentMetric"])

engine_concat = search_engine("concat", store, client)
print(engine_concat.search(keyword="매출 개선", top_k=5))

engine_per_key = search_engine("per_key", store, client)
print(
    engine_per_key.search(
        keyword_by_field={"contentNm": "매출", "contentMetric": "전환율"},
        weight_by_field={"contentNm": 0.7, "contentMetric": 0.3},
        top_k=5,
    )
)
```

## 3) MongoDB CRUD (id 기준)

```python
from pymongo import MongoClient
from simple_search import EmbeddingClient, search_store

mongo = MongoClient("mongodb://localhost:27017")
col = mongo["search_db"]["contents"]

store = search_store(EmbeddingClient(api_url="mock", provider="mock"), id_key="contentId")

store.create_content(col, {"contentId": "C001", "contentNm": "매출 개선"}, upsert=True)
item = store.read_content(col, "C001")
store.update_content(col, "C001", {"contentDesc": "설명 업데이트"})
store.delete_content(col, "C001")
```

## 4) MongoDB 분리 저장/로드 (metadata + embedding)

```python
from pymongo import MongoClient
from simple_search import EmbeddingClient, search_store

mongo = MongoClient("mongodb://localhost:27017")
meta_col = mongo["search_db"]["content_metadata"]
emb_col = mongo["search_db"]["content_embeddings"]

store = search_store(EmbeddingClient(api_url="mock", provider="mock"), id_key="contentId")
store.load_contents_from_json("sample_contents.json")
store.build_embeddings(mode="both")

# split 저장
store.save_to_mongodb_split(meta_col, emb_col, upsert=True)

# split 로드
store.load_from_mongodb_split(meta_col, emb_col)
```

### 저장 문서 형태 예시

`contentId = "C001"`인 경우:

- metadata collection:

```json
{
  "contentId": "C001",
  "contentNm": "매출 개선 전략",
  "contentDesc": "전환율 향상을 위한 마케팅 캠페인"
}
```

- embedding collection (`mode="concat"`):

```json
{
  "contentId": "C001",
  "concat_embedding": [0.12, -0.03, 0.55, -0.21]
}
```

- embedding collection (`mode="per_key"`):

```json
{
  "contentId": "C001",
  "field_embeddings": {
    "contentNm": [0.01, 0.12, -0.05, 0.89],
    "contentDesc": [0.45, -0.11, 0.03, 0.22]
  }
}
```

- embedding collection (`mode="both"`):

```json
{
  "contentId": "C001",
  "concat_embedding": [0.12, -0.03, 0.55, -0.21],
  "field_embeddings": {
    "contentNm": [0.01, 0.12, -0.05, 0.89],
    "contentDesc": [0.45, -0.11, 0.03, 0.22]
  }
}
```
