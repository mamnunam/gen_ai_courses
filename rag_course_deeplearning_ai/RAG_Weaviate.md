# Weaviate — Vector Database

## What Weaviate Does

Weaviate is an open-source vector database that handles the full embedding-and-search workflow in one system:

1. Receives objects with text properties
2. Automatically embeds them using a configured model
3. Indexes the vectors using HNSW (approximate nearest-neighbor)
4. Exposes four search modes: metadata filter, semantic (near_text), BM25, hybrid

---

## 1. Connecting

### Embedded (in-process, local)

```python
import weaviate

client = weaviate.connect_to_embedded(
    persistence_data_path="./.collections",  # where to store data across restarts
    environment_variables={
        "ENABLE_MODULES":              "text2vec-transformers, reranker-transformers",
        "TRANSFORMERS_INFERENCE_API":  "http://127.0.0.1:5000/",
        "RERANKER_INFERENCE_API":      "http://127.0.0.1:5000/"
    }
)
```

### Remote / already-running instance

```python
client = weaviate.connect_to_local(port=8079, grpc_port=50050)

# Always close when done
client.close()
```

---

## 2. Creating a Collection

A **collection** is Weaviate's term for a group of objects sharing a schema. Think of it like a typed table + a vector index.

```python
from weaviate.classes.config import Configure, Property, DataType, Tokenization

# Configure the embedding model
vectorizer_config = [
    Configure.NamedVectors.text2vec_transformers(
        name="main_vector",                              # name to access vectors later
        source_properties=["title", "chunk", "description"],  # properties to embed (concatenated)
        vectorize_collection_name=False,                 # don't prepend collection name to text
        inference_url="http://127.0.0.1:5000"
    )
]

if not client.collections.exists("bbc_collection"):
    collection = client.collections.create(
        name="bbc_collection",
        vectorizer_config=vectorizer_config,
        reranker_config=Configure.Reranker.transformers(),  # enable reranking
        properties=[
            Property(name="title",         data_type=DataType.TEXT),
            Property(name="chunk",         data_type=DataType.TEXT),
            Property(name="chunk_index",   data_type=DataType.INT),
            Property(name="description",   data_type=DataType.TEXT),
            Property(name="link",          data_type=DataType.TEXT),
            Property(name="pubDate",       data_type=DataType.DATE),
            # FIELD tokenization means the whole string is treated as one token (exact match)
            Property(name="chunking_strategy", data_type=DataType.TEXT,
                     tokenization=Tokenization.FIELD),
        ]
    )
else:
    collection = client.collections.get("bbc_collection")
```

---

## 3. Inserting Data

```python
from weaviate.util import generate_uuid5
from tqdm import tqdm

# batch.fixed_size: send batch_size objects per API call
with collection.batch.fixed_size(batch_size=20, concurrent_requests=5) as batch:
    for doc in tqdm(data):
        # generate_uuid5 creates a deterministic ID from the content
        # → re-running the insert won't create duplicate entries
        uuid = generate_uuid5(doc)
        batch.add_object(properties=doc, uuid=uuid)

print(f"Collection size: {len(collection)}")
```

---

## 4. Inspecting Objects

```python
# Fetch a single object with its vector
obj = collection.query.fetch_objects(limit=1, include_vector=True).objects[0]

print(obj.properties)                         # dict of all property values
print(obj.vector['main_vector'][:10])         # first 10 dims of the embedding
print(len(obj.vector['main_vector']))         # 768
```

---

## 5. Query Mode 1 — Metadata Filtering

Filter by property value without vector search. Uses exact match or range operators.

```python
from weaviate.classes.query import Filter

# Equal
result = collection.query.fetch_objects(
    filters=Filter.by_property("budget").equal("Low"),
    limit=5
)

# Contains any value from a list
result = collection.query.fetch_objects(
    filters=Filter.by_property("title").contains_any(["Taylor Swift"]),
    limit=2
)

# Numeric range
result = collection.query.fetch_objects(
    filters=Filter.by_property("user_ratings").greater_or_equal(3.5),
    limit=5
)

# Combine filters with AND
result = collection.query.near_text(
    query="casual outfit",
    filters=Filter.all_of([
        Filter.by_property("gender").contains_any(["Men"]),
        Filter.by_property("season").contains_any(["Summer"]),
    ]),
    limit=10
)

# Extract properties
objects = [x.properties for x in result.objects]
```

### RAG helper function

```python
def filter_by_metadata(
    property_name: str,
    values: list,
    collection,
    limit: int = 5
) -> list[dict]:
    response = collection.query.fetch_objects(
        filters=Filter.by_property(property_name).contains_any(values),
        limit=limit
    )
    return [x.properties for x in response.objects]
```

---

## 6. Query Mode 2 — Semantic Search (near_text)

Embed the query and return the most similar objects by cosine distance.

```python
def semantic_search_retrieve(query: str, collection, top_k: int = 5) -> list[dict]:
    response = collection.query.near_text(query=query, limit=top_k)
    return [x.properties for x in response.objects]

# With metadata filter combined
response = collection.query.near_text(
    query="cheap places to visit in winter",
    filters=Filter.by_property("budget").contains_any(["Low", "Moderate"]),
    limit=5
)
```

---

## 7. Query Mode 3 — BM25 Keyword Search

```python
def bm25_retrieve(query: str, collection, top_k: int = 5) -> list[dict]:
    response = collection.query.bm25(query=query, limit=top_k)
    return [x.properties for x in response.objects]

# With filter
response = collection.query.bm25(
    query="remote repository git",
    filters=Filter.by_property("chapter_title").equal("02-git-basics"),
    limit=5
)
```

---

## 8. Query Mode 4 — Hybrid Search (RRF)

Weaviate implements hybrid search internally using RRF. The `alpha` parameter blends BM25 and vector results:

- `alpha = 0.0` → pure BM25
- `alpha = 1.0` → pure vector
- `alpha = 0.5` → equal blend (good default)

```python
def hybrid_retrieve(
    query: str,
    collection,
    alpha: float = 0.5,
    top_k: int = 5
) -> list[dict]:
    response = collection.query.hybrid(query=query, alpha=alpha, limit=top_k)
    return [x.properties for x in response.objects]

# With filter and custom alpha
response = collection.query.hybrid(
    query="cheap winter travel",
    filters=Filter.by_property("budget").contains_any(["Low", "Moderate"]),
    alpha=0.3,   # lean more towards BM25
    limit=5
)
```

---

## 9. Reranking

Reranking adds a second-pass cross-attention model that re-scores the retrieved results based on a specified property and query. This is more expensive but often improves ordering significantly.

```python
from weaviate.classes.query import Rerank

def semantic_search_with_reranking(
    query: str,
    rerank_property: str,       # which property to score against
    collection,
    rerank_query: str = None,   # if None, uses the original query
    top_k: int = 5
) -> list[dict]:
    if rerank_query is None:
        rerank_query = query

    response = collection.query.near_text(
        query=query,
        limit=top_k,
        rerank=Rerank(prop=rerank_property, query=rerank_query)
    )
    return [x.properties for x in response.objects]

# Example: retrieve by full semantic similarity, rerank specifically on chunk text
results = semantic_search_with_reranking(
    query="conflicts in Latin America",
    rerank_property="chunk",
    collection=collection,
    top_k=5
)

# Example: use a different query for reranking
results = semantic_search_with_reranking(
    query="Tell me about Taylor Swift's Eras Tour",
    rerank_property="chunk",
    rerank_query="Wembley record attendance",   # more specific for reranking
    collection=collection,
    top_k=5
)
```

---

## 10. Aggregation & Collection Management

```python
# Count all objects
len(collection)

# Count with filter
count = collection.aggregate.over_all(
    filters=Filter.by_property("chunking_strategy").equal("fixed_size_25")
).total_count

# List all collections
client.collections.list_all().keys()

# Check if a collection exists
client.collections.exists("my_collection")

# Delete a collection (irreversible!)
client.collections.delete("my_collection")
```

---

## 11. Weaviate in the RAG Pipeline

```python
def generate_final_prompt(
    query: str,
    top_k: int,
    retrieve_function,
    collection,
    use_rag: bool = True
) -> str:
    if not use_rag:
        return query

    docs = retrieve_function(query=query, collection=collection, top_k=top_k)

    context = "".join([
        f"Title: {d['title']}, Chunk: {d['chunk']}, "
        f"Published: {d['pubDate']}\nURL: {d['link']}\n"
        for d in docs
    ])

    return (
        f"Answer the query using the context below. "
        f"Context is ordered by relevance.\n"
        f"Query: {query}\nContext:\n{context}"
    )

def llm_call(query, retrieve_function=None, top_k=5, use_rag=True, collection=None):
    prompt   = generate_final_prompt(query, top_k, retrieve_function, collection, use_rag)
    response = generate_with_single_input(prompt)
    return response['content']

# Usage
answer = llm_call("What happened at Wembley?", retrieve_function=hybrid_retrieve,
                  collection=collection)
```

---

## 12. Common Patterns Summary

| Task | Method |
|------|--------|
| Find articles by exact title keywords | `bm25_retrieve` |
| Find semantically similar content | `semantic_search_retrieve` |
| Best general-purpose search | `hybrid_retrieve` (alpha=0.5) |
| Filter by category/date/price then search | `near_text` + `Filter.all_of(...)` |
| Re-order noisy initial results | `semantic_search_with_reranking` |
| Count documents matching a condition | `collection.aggregate.over_all(filters=...)` |

---

## 13. ANN Algorithms Under the Hood (HNSW)

Weaviate uses **HNSW** (Hierarchical Navigable Small World) for approximate nearest-neighbor search. To understand why, walk up the algorithmic ladder:

### KNN baseline (exact)

```
For each query:
  Compute distance to every document  →  O(N) per query
  Sort                                 →  O(N log N)
```

With 1 billion documents that's a billion distance calculations *per search*. Linear scaling means no real-world system can use exact KNN at scale.

### Navigable Small World (NSW)

Build a **proximity graph** — each document is a node, connected to its nearest neighbors. To find the closest doc to a query:

1. Start at a random node.
2. Look at neighbors; jump to the one closest to the query.
3. Repeat until no neighbor is closer (local minimum).

Sublinear, but greedy search can miss the global optimum.

### HNSW — the production answer

Stack multiple NSW graphs at decreasing sparsity:

```
Layer 3 (~10 vectors)     ◄── coarse, fast top-level jumps
Layer 2 (~100 vectors)    ◄── intermediate
Layer 1 (all ~1,000)      ◄── precise final search
```

Search descends layer by layer, using the best candidate from each layer as the entry point for the next. **Scaling is approximately logarithmic** in N — billions of vectors searchable in milliseconds.

**Build vs query tradeoff:** building the index is expensive (relatively), but it's done once and reused for every query. Weaviate builds the index incrementally as you insert objects.

---

## 14. Pre-Built Collection Pattern

Production code often consumes a collection that was created and populated elsewhere. The pattern:

```python
# Connect to a Weaviate instance already running (e.g., a Flask app + weaviate_server)
client = weaviate.connect_to_local(port=8079, grpc_port=50050)

# Just grab the existing collection — don't create or populate
products_collection = client.collections.get('products')
print(len(products_collection))  # __len__ supported

# Use directly
results = products_collection.query.near_text("blue summer dress", limit=10)
```

### Killing stale processes before reconnecting

When iterating in notebooks, leftover Weaviate or Flask processes block restarts:

```python
from utils import kill_processes_on_ports
kill_processes_on_ports([5000, 8080, 8097, 50050, 50051])

# Now safe to launch fresh
from utils import flask_app, weaviate_server
```

---

## 15. Filter Operator Cheat Sheet

| Operator | Use case | Example |
|----------|---------|---------|
| `.equal(v)` | Exact match on text/number/date | `Filter.by_property("gender").equal("Men")` |
| `.not_equal(v)` | Exclusion | `Filter.by_property("status").not_equal("draft")` |
| `.contains_any([...])` | Multi-value OR match | `Filter.by_property("color").contains_any(["Red","Blue"])` |
| `.contains_all([...])` | Multi-value AND match | Tags that include both "summer" and "casual" |
| `.greater_than(v)` / `.greater_or_equal(v)` | Numeric/date range | `Filter.by_property("price").greater_than(50)` |
| `.less_than(v)` / `.less_or_equal(v)` | Numeric/date range | `Filter.by_property("price").less_than(300)` |
| `.like("pattern*")` | Wildcard text match | `Filter.by_property("name").like("Levi*")` |
| `.is_null(True)` | Missing values | `Filter.by_property("description").is_null(True)` |
| `Filter.all_of([...])` | AND combine | `Filter.all_of([f1, f2, f3])` |
| `Filter.any_of([...])` | OR combine | `Filter.any_of([f1, f2])` |

### Tokenization choice affects BM25 behavior

```python
Property(name="category", data_type=DataType.TEXT, tokenization=Tokenization.FIELD)
```

`Tokenization.FIELD` treats the whole property value as a single token. Critical for category/tag fields where you want exact-string BM25 matches (e.g., `chunking_strategy="fixed_size_25"` shouldn't split on the underscores). Default tokenization is `WORD`.

---

## 16. Distance Metrics in Weaviate

Configured at collection creation time:

```python
from weaviate.classes.config import VectorDistances

vectorizer_config = [Configure.NamedVectors.text2vec_transformers(
    name="vector",
    source_properties=["title", "chunk"],
    vector_index_config=Configure.VectorIndex.hnsw(
        distance_metric=VectorDistances.COSINE   # COSINE, DOT, L2_SQUARED, HAMMING, MANHATTAN
    )
)]
```

For text embeddings, **cosine** is the standard. Use `DOT` if your model was trained for it (some OpenAI embeddings); `L2_SQUARED` (Euclidean) is rare for text.

---

## See Also

- [[RAG_Chunking]] — preparing documents before loading into Weaviate
- [[RAG_Retrieval_Methods]] — the same BM25/semantic/RRF concepts implemented from scratch
- [[RAG_Prompt_Engineering]] — routing queries to different Weaviate search modes
- [[RAG_Advanced_Retrieval]] — what cross-encoder reranking does under the hood
