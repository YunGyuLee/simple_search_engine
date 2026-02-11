# Simple Search Engine (Python 3.11)

요청사항 기준으로 구조를 단순화했습니다.

- `EmbeddingClient`: 외부 임베딩 API 호출
- `search_store`: 
  - ID 기준 MongoDB CRUD
  - 벌크 load/save
  - 단건/벌크 embedding
  - 검색용 인메모리 임베딩 인덱스 생성
- `search_engine`: 코사인 유사도 검색

> 참고: mock 임베딩 코드는 제거했습니다.

## 1) 설치

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) 기본 사용법

```python
from simple_search import EmbeddingClient, search_store, search_engine

embedding_client = EmbeddingClient(
    api_url="https://your-embedding-api/v1/embeddings",
    api_key="YOUR_API_KEY",
    model="text-embedding-model",
    timeout=10,
)

store = search_store(embedding_client=embedding_client, id_key="contentId")
store.load_contents_from_json("sample_contents.json")
store.build_embeddings(
    mode="both",
    concat_keys=["contentNm", "contentDesc"],
    per_key_fields=["contentNm", "contentDesc", "contentMetric"],
)

engine = search_engine("concat", store, embedding_client)
print(engine.search(keyword="매출 개선", top_k=5))
```

## 3) search_store 핵심 구조

### A. ID 기준 CRUD

```python
from pymongo import MongoClient
from simple_search import EmbeddingClient, search_store

mongo = MongoClient("mongodb://localhost:27017")
col = mongo["search_db"]["contents"]

store = search_store(EmbeddingClient(api_url="https://your-embedding-api/v1/embeddings"), id_key="contentId")

store.create_content(col, {"contentId": "C001", "contentNm": "매출 개선"}, upsert=True)
item = store.read_content(col, "C001")
store.update_content(col, "C001", {"contentDesc": "설명 업데이트"})
store.delete_content(col, "C001")
```

### B. 벌크 load/save

```python
store.load_contents_from_mongodb(col, query={"category": "A"}, limit=1000)
store.save_contents_to_mongodb(col, upsert=True)
```

### C. embedding 메소드

- 인메모리 인덱스 생성: `build_embeddings(...)`
- 단건 DB 임베딩 저장: `embed_content_by_id(...)`
- 벌크 DB 임베딩 저장: `embed_contents_bulk(...)`

```python
# 단건
payload = store.embed_content_by_id(
    collection=col,
    content_id="C001",
    concat_keys=["contentNm", "contentDesc"],
    per_key_fields=["contentNm", "contentDesc", "contentMetric"],
    embedding_field="embeddings",
)

# 벌크
updated_count = store.embed_contents_bulk(
    collection=col,
    query={"status": "active"},
    limit=500,
    concat_keys=["contentNm", "contentDesc"],
    per_key_fields=["contentNm", "contentDesc", "contentMetric"],
    embedding_field="embeddings",
)
print(updated_count)
```

`embedding_field="embeddings"` 기준 저장 예시:

```json
{
  "contentId": "C001",
  "contentNm": "매출 개선 전략",
  "embeddings": {
    "concat": [0.12, -0.03, 0.55, -0.21],
    "per_key": {
      "contentNm": [0.01, 0.12, -0.05, 0.89],
      "contentDesc": [0.45, -0.11, 0.03, 0.22]
    }
  }
}
```
