# Simple Search Engine (Python 3.11)

요청사항 기준 구조:

- `EmbeddingClient`: 외부 임베딩 API 호출
- `search_store`:
  - ID 기준 MongoDB CRUD
  - 벌크 load/save
  - 단건/벌크 embedding
  - 검색용 인메모리 임베딩 인덱스 관리
  - **이미 저장된 임베딩 재사용(load/cache) 지원**
- `search_engine`: 코사인 유사도 검색

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

# reuse_cached_embeddings=True (기본): 문서에 embeddings가 이미 있으면 우선 사용
store.build_embeddings(
    mode="both",
    concat_keys=["contentNm", "contentDesc"],
    per_key_fields=["contentNm", "contentDesc", "contentMetric"],
    reuse_cached_embeddings=True,
    embedding_field="embeddings",
)

engine = search_engine("concat", store, embedding_client)
print(engine.search(keyword="매출 개선", top_k=5))
```

## 3) MongoDB 벌크 로드 시 기존 임베딩 로드

```python
# load_embeddings_if_present=True (기본)
# Mongo 문서에 embeddings 필드가 있으면 인메모리 임베딩 캐시로 복원 시도
store.load_contents_from_mongodb(
    collection=col,
    query={"status": "active"},
    limit=1000,
    load_embeddings_if_present=True,
    embedding_field="embeddings",
)
```

## 4) search_store API 구조

### A. ID 기준 CRUD

- `create_content(...)`
- `read_content(...)`
- `update_content(...)`
- `delete_content(...)`

### B. 벌크 load/save

- `load_contents_from_json(...)`
- `load_contents_from_mongodb(...)`
- `save_contents_to_mongodb(...)`
- `replace_contents(...)`

### C. embedding

- 검색용 인메모리 캐시 생성: `build_embeddings(...)`
- 단건 DB 임베딩 저장: `embed_content_by_id(...)`
- 벌크 DB 임베딩 저장: `embed_contents_bulk(...)`
- 문서 내 저장 임베딩 복원: `load_cached_embeddings_from_contents(...)`

## 5) 임베딩 저장 형태 (단일 collection)

`embedding_field="embeddings"`일 때 문서 예시:

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
