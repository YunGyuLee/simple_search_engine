# Simple Search Engine (Python 3.11)

요청사항 기준 구조:

- `EmbeddingClient`: 외부 임베딩 API 호출
- `search_store`:
  - ID 기준 MongoDB CRUD
  - 벌크 load/save
  - 단건/벌크 embedding
  - 검색용 인메모리 임베딩 인덱스 관리
  - 이미 저장된 임베딩 재사용(load/cache)
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

client = EmbeddingClient(
    api_url="https://your-embedding-api/v1/embeddings",
    api_key="YOUR_API_KEY",
    model="text-embedding-model",
)

store = search_store(client, id_key="contentId")
store.load_contents_from_json("sample_contents.json")

store.build_embeddings(
    mode="both",
    concat_keys=["contentNm", "contentDesc"],
    per_key_fields=["contentNm", "contentDesc", "contentMetric"],
    reuse_cached_embeddings=True,
    embedding_field="embeddings",
)

engine = search_engine("concat", store, client)
print(engine.search(keyword="매출 개선", top_k=5))
```

## 3) MongoDB 로드 시 기존 임베딩 재사용

```python
store.load_contents_from_mongodb(
    collection=col,
    query={"status": "active"},
    limit=1000,
    load_embeddings_if_present=True,
    embedding_field="embeddings",
)

# 캐시가 충분하면 API 재호출 없이 사용
store.build_embeddings(mode="concat", reuse_cached_embeddings=True, embedding_field="embeddings")
```

## 4) concat 조합 key 정보 저장

`embed_content_by_id` / `embed_contents_bulk`로 저장할 때,
concat 임베딩에 어떤 key 조합을 사용했는지 저장하며, 필요 시 여러 조합을 `concat_by_combo`로 함께 저장할 수 있습니다.

```json
{
  "contentId": "C001",
  "embeddings": {
    "concat": [0.12, -0.03, 0.55, -0.21],
    "concat_meta": {
      "keys": ["contentNm", "contentDesc"]
    },
    "concat_by_combo": {
      "default": {
        "keys": ["contentNm", "contentDesc"],
        "vector": [0.12, -0.03, 0.55, -0.21]
      },
      "nm_period": {
        "keys": ["contentNm", "contentPeriod"],
        "vector": [0.05, 0.77, -0.11, 0.21]
      }
    },
    "per_key": {
      "contentNm": [0.01, 0.12, -0.05, 0.89],
      "contentDesc": [0.45, -0.11, 0.03, 0.22]
    }
  }
}
```


여러 조합 저장 예시:

```python
store.embed_contents_bulk(
    collection=col,
    concat_key_groups={
        "default": ["contentNm", "contentDesc"],
        "nm_period": ["contentNm", "contentPeriod"],
    },
    embedding_field="embeddings",
)
```

## 5) search_store API 요약

- CRUD: `create_content`, `read_content`, `update_content`, `delete_content`
- 벌크: `load_contents_from_json`, `load_contents_from_mongodb`, `save_contents_to_mongodb`, `replace_contents`
- 임베딩: `build_embeddings`, `embed_content_by_id`, `embed_contents_bulk`, `load_cached_embeddings_from_contents`
