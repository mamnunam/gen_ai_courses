# Retrieval Methods

Three retrieval approaches are covered in this course, each with different strengths. In practice, **hybrid search (RRF)** is the best default.

| Method | Type | Strengths | Weaknesses |
|--------|------|-----------|------------|
| BM25 | Sparse / keyword | Fast, exact keyword match | No semantic understanding |
| Semantic search | Dense / embedding | Understands meaning, synonyms | Can miss exact terms |
| RRF (hybrid) | Both combined | Best of both worlds | Slightly more complex |

---

## 1. BM25

BM25 is a classic probabilistic algorithm that scores documents based on:
- **Term frequency** — how often query terms appear in the document
- **Inverse document frequency** — how rare/unique those terms are across the whole corpus
- **Document length normalization** — penalizes unusually long documents

### Library: `bm25s`

```python
import bm25s

# Build the corpus string (one entry per document)
corpus = [x['title'] + " " + x['description'] for x in NEWS_DATA]

# Index the corpus
BM25_RETRIEVER = bm25s.BM25(corpus=corpus)
TOKENIZED_DATA = bm25s.tokenize(corpus)
BM25_RETRIEVER.index(TOKENIZED_DATA)

# Retrieve
def bm25_retrieve(query: str, top_k: int = 5) -> list[int]:
    tokenized_query = bm25s.tokenize(query)
    results, scores = BM25_RETRIEVER.retrieve(tokenized_query, k=top_k)
    results = results[0]                              # unwrap batch dimension
    return [corpus.index(doc) for doc in results]    # convert docs → indices

bm25_retrieve("What are the recent news about GDP?")
# → [752, 673, 289, 626, 43]
```

**When BM25 wins:** Queries with rare or specific keywords ("GDPR", "Wembley", a person's name).  
**When BM25 loses:** Paraphrase queries — "automobile industry" won't match documents about "car manufacturing".

---

## 2. Semantic Search

Semantic search encodes both the query and all documents as embedding vectors, then ranks by cosine similarity. See [[RAG_Embeddings]] for the underlying math.

```python
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

model      = SentenceTransformer("BAAI/bge-base-en-v1.5")
EMBEDDINGS = joblib.load("embeddings.joblib")   # shape: (N, 768), pre-computed

def cosine_similarity(v1, array_of_vectors):
    v1 = np.array(v1, dtype=np.float32).ravel()
    A  = np.atleast_2d(np.array(array_of_vectors, dtype=np.float32))
    v1_norm = np.linalg.norm(v1)
    A_norms = np.linalg.norm(A, axis=1)
    denom   = v1_norm * A_norms
    sims    = (A @ v1) / np.where(denom == 0, 1.0, denom)
    return sims.tolist()

def semantic_search_retrieve(query: str, top_k: int = 5) -> list[int]:
    query_emb = model.encode(query)
    scores    = cosine_similarity(query_emb, EMBEDDINGS)
    indices   = np.argsort(-np.array(scores))
    return [int(i) for i in indices[:top_k]]

semantic_search_retrieve("What are the recent news about GDP?")
# → [743, 673, 626, 752, 326]
```

**When semantic search wins:** Paraphrase queries, conceptual questions, when exact keywords are unlikely.  
**When semantic search loses:** Queries containing rare proper nouns or very specific technical terms that may not be well-represented in the embedding space.

---

## 3. Reciprocal Rank Fusion (RRF)

RRF combines ranked lists from multiple retrievers. Each document gets a score based on its position in each list; scores are summed across lists.

### Formula

$$\text{Score}(d) = \sum_{r=1}^{n} \frac{1}{k + \text{rank}_r(d)}$$

- $k$ = smoothing constant (default: 60) — prevents very high scores for rank-1 documents
- $\text{rank}_r(d)$ = 1-indexed position of document $d$ in list $r$
- Documents appearing high in **multiple** lists accumulate the highest scores

### Why this works

A document ranked #1 in semantic search but absent from BM25 is less trustworthy than a document ranked #3 in both. RRF rewards consistent multi-system relevance.

```python
def reciprocal_rank_fusion(
    list1: list[int],
    list2: list[int],
    top_k: int = 5,
    K: int = 60
) -> list[int]:
    scores = {}
    for lst in [list1, list2]:
        for rank, doc_id in enumerate(lst, start=1):   # ranks are 1-indexed
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + K)
    
    sorted_docs = sorted(scores, key=scores.get, reverse=True)
    return [int(d) for d in sorted_docs[:top_k]]

# Example
list1 = semantic_search_retrieve("What are the recent news about GDP?")  # [743, 673, 626, 752, 326]
list2 = bm25_retrieve("What are the recent news about GDP?")              # [752, 673, 289, 626,  43]

rrf_list = reciprocal_rank_fusion(list1, list2)
# → [673, 752, 626, 743, 289]
# doc 673 appears in both lists at rank 2 → gets the highest combined score
```

---

## 4. Full RAG Pipeline with Switchable Retrievers

```python
def generate_final_prompt(
    query: str,
    top_k: int = 5,
    retrieve_function=None,
    use_rag: bool = True
) -> str:
    if not use_rag:
        return query

    # Handle RRF (needs two sub-retrievers)
    if retrieve_function.__name__ == "reciprocal_rank_fusion":
        list1       = semantic_search_retrieve(query, top_k)
        list2       = bm25_retrieve(query, top_k)
        top_indices = retrieve_function(list1, list2, top_k)
    else:
        top_indices = retrieve_function(query=query, top_k=top_k)

    docs = query_corpus(top_indices)
    context = "\n".join([
        f"Title: {d['title']}, Description: {d['description']}, "
        f"Published: {d['published_at']}\nURL: {d['url']}"
        for d in docs
    ])

    return (
        f"Answer the query using the context below. "
        f"Context is from 2024 and supplements your existing knowledge.\n"
        f"Query: {query}\nContext:\n{context}"
    )

def llm_call(query, retrieve_function=None, top_k=5, use_rag=True):
    prompt   = generate_final_prompt(query, top_k, retrieve_function, use_rag)
    response = generate_with_single_input(prompt)
    return response['content']

# Usage
llm_call("Recent AI advances.", retrieve_function=bm25_retrieve)
llm_call("Recent AI advances.", retrieve_function=semantic_search_retrieve)
llm_call("Recent AI advances.", retrieve_function=reciprocal_rank_fusion)
llm_call("Recent AI advances.", use_rag=False)   # baseline: no retrieval
```

---

## 5. Evaluation Metrics

To evaluate a retriever you need a **labeled dataset** — queries where the correct relevant documents are known.

### Precision@K

What fraction of the K retrieved documents are actually relevant?

$$\text{Precision@K} = \frac{\text{# relevant in top-K}}{K}$$

```python
def precision_at_k(relevant_count: int, k: int) -> float:
    if k == 0:
        return 0.0
    return relevant_count / k
```

### Recall@K

What fraction of *all* relevant documents in the corpus did we retrieve in our top-K?

$$\text{Recall@K} = \frac{\text{# relevant in top-K}}{\text{total relevant in corpus}}$$

```python
def recall_at_k(relevant_count: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    return relevant_count / total_relevant
```

### The Precision-Recall Tradeoff

As K increases, you retrieve more of the relevant documents (recall goes up), but include more irrelevant ones (precision goes down).

| K | Precision | Recall |
|---|-----------|--------|
| 5 | ~1.00 (very high) | ~0.01 (very low) |
| 20 | ~0.80 | ~0.03 |
| 50 | ~0.65 | ~0.08 |

### Evaluating over a query set

```python
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)

test_queries = [
    {"query": "advancements in space exploration", "desired_category": "sci.space"},
    {"query": "real-time rendering in computer graphics", "desired_category": "comp.graphics"},
    {"query": "NHL playoffs statistics",              "desired_category": "rec.sport.hockey"},
    # ...
]

def compute_metrics(queries, embeddings, model, top_k=5):
    results = []
    E = np.vstack([np.asarray(x, dtype=np.float32).ravel() for x in embeddings])

    for item in queries:
        q_emb  = model.encode(item["query"]).astype(np.float32)
        scores = cosine_similarity(q_emb, E)
        top_k_idx = np.argsort(-np.array(scores))[:top_k]

        retrieved_categories = [
            newsgroups.target_names[df.iloc[i]["category"]] for i in top_k_idx
        ]
        relevant_in_top_k   = sum(1 for c in retrieved_categories if c == item["desired_category"])
        total_relevant       = sum(1 for i in range(len(df))
                                   if newsgroups.target_names[df.iloc[i]["category"]] == item["desired_category"])

        results.append({
            "query":      item["query"],
            "precision@k": precision_at_k(relevant_in_top_k, top_k),
            "recall@k":    recall_at_k(relevant_in_top_k, total_relevant),
        })
    return results
```

### Practical guidance for RAG

- **K = 5–20** is the typical sweet spot: high precision (context isn't polluted) + manageable prompt length.
- You don't need *all* relevant documents — just enough good ones. So recall@5 being low is acceptable.
- If your eval shows precision dropping sharply at K=10, your retriever is noisy — consider reranking.

---

## 6. Retriever Comparison Summary

| | BM25 | Semantic | RRF |
|--|------|----------|-----|
| Handles synonyms | ❌ | ✅ | ✅ |
| Exact keyword match | ✅ | ⚠️ | ✅ |
| Needs embedding model | ❌ | ✅ | ✅ |
| Needs pre-computed embeddings | ❌ | ✅ | ✅ |
| Speed | Very fast | Medium | Medium |
| Best default | — | — | ✅ |

---

## See Also

- [[RAG_Embeddings]] — how embeddings work under the hood
- [[RAG_Weaviate]] — BM25, semantic, and hybrid search via Weaviate API
- [[RAG_Prompt_Engineering]] — routing to different retrievers based on query type
