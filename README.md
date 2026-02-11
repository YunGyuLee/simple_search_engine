# Simple Search Engine (Python 3.11)

요청사항 기준 구조:

- `EmbeddingClient`: 외부 임베딩 API 호출
- `search_store`:
  - ID 기준 MongoDB CRUD
  - 벌크 load/save
  - 단건/벌크 embedding
  - 검색용 인메모리 임베딩 인덱스 관리
  - 이미 저장된 임베딩 재사용(load/cache)
  - **여러 concat 조합 지원 + 조합 정보(keys) 저장**
- `search_engine`: 코사인 유사도 검색 (`combo_name` 지정 가능)

## 1) 설치

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) 여러 concat 조합으로 임베딩 인덱스 생성

```python
from simple_search import EmbeddingClient, search_store

client = EmbeddingClient(
    api_url="https://your-embedding-api/v1/embeddings",
    api_key="YOUR_API_KEY",
    model="text-embedding-model",
)

store = search_store(client, id_key="contentId")
store.load_contents_from_json("sample_contents.json")

store.build_embeddings(
    mode="both",
    concat_key_groups={
        "name_desc": ["contentNm", "contentDesc"],
        "goal_action": ["contentObject", "contentAction"],
        "full": ["contentNm", "contentDesc", "contentObject", "contentAction", "contentMetric"],
    },
    per_key_fields=["contentNm", "contentDesc", "contentMetric"],
    reuse_cached_embeddings=True,
    embedding_field="embeddings",
)
```

## 3) concat 조합 지정 검색

```python
from simple_search import search_engine

engine = search_engine("concat", store, client)

# 조합 지정
r1 = engine.search(keyword="매출 개선", combo_name="name_desc", top_k=5)

# 조합 미지정 -> default 조합 사용
r2 = engine.search(keyword="매출 개선", top_k=5)
```

## 4) MongoDB 로드 시 기존 임베딩 재사용

```python
store.load_contents_from_mongodb(
    collection=col,
    query={"status": "active"},
    limit=1000,
    load_embeddings_if_present=True,
    embedding_field="embeddings",
)

# 필요 캐시가 충분하면 API 재호출 없이 사용
store.build_embeddings(mode="concat", reuse_cached_embeddings=True, embedding_field="embeddings")
```

## 5) 임베딩 저장 형식 (조합 정보 포함)

`embed_content_by_id` / `embed_contents_bulk`는 아래 형식으로 저장합니다.

```json
{
  "contentId": "C001",
  "embeddings": {
    "concat": [0.12, -0.03, 0.55, -0.21],
    "concat_by_combo": {
      "name_desc": {
        "keys": ["contentNm", "contentDesc"],
        "vector": [0.12, -0.03, 0.55, -0.21]
      },
      "goal_action": {
        "keys": ["contentObject", "contentAction"],
        "vector": [0.09, 0.11, -0.30, 0.05]
      },
      "default": {
        "keys": ["contentNm", "contentDesc"],
        "vector": [0.12, -0.03, 0.55, -0.21]
      }
    },
    "per_key": {
      "contentNm": [0.01, 0.12, -0.05, 0.89],
      "contentDesc": [0.45, -0.11, 0.03, 0.22]
    }
  }
}
```

`concat_by_combo.<combo_name>.keys`에 어떤 키 조합으로 concat 임베딩을 만들었는지가 저장됩니다.
