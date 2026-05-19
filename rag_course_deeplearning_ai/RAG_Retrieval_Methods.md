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

## How BM25 Got Here — TF-IDF Derivation

BM25 is the 25th refinement in the "Best Matching" series. Conceptually it descends from TF-IDF, walked through in stages:

### Step 1 — Simple scoring
Count how many query words appear in the document. *Problem:* doesn't reward frequent occurrences.

### Step 2 — Term Frequency (TF)
Sum frequencies of each query word in the document. *Problem:* favors long documents (more words → more matches by chance).

### Step 3 — Normalized TF
Divide by total words in the document. *Problem:* common words ("the", "and") dominate.

### Step 4 — TF-IDF
Multiply TF by **inverse document frequency**:

$$\text{IDF}(\text{word}) = \log \frac{\text{Total docs}}{\text{Docs containing word}}$$

Rare words get high IDF; common words get low IDF. The query *"Making pizza without a pizza oven"* over a 5-doc corpus produces high IDF for `pizza` (in 2/5 → ~0.7) and very low for `a` (in 4/5 → ~0.1). Pizza and oven dominate the score.

### Step 5 — BM25's two refinements

1. **Term frequency saturation** (parameter `k₁`, typical 1.2–2.0): diminishing returns on repeated terms. If `pizza` appearing 10 times scores X, then 20 times scores ~1.3X (not 2X). Prevents one keyword-stuffing document from dominating.
2. **Document length normalization** (parameter `b`, typical 0.75): penalizes long documents *less* than raw TF-IDF would. With `b=1` you get full normalization; `b=0` you get none.

### Bag-of-words view

BM25 treats each document as a **sparse vector** in vocabulary space — each dimension is a vocabulary term, most dimensions are zero, the rest hold (weighted) frequencies. Visualized as a doc-term matrix where columns are documents and rows are vocabulary words. Hence the name **sparse retrieval**.

---

## More on RRF — Why It Avoids Score Normalization

RRF only cares about **ranks**, not raw scores. This matters because:

- BM25 scores are unbounded and corpus-dependent (one corpus's `0.5` is another's `5.0`).
- Cosine similarities are bounded `[−1, 1]`.
- Naive linear combination would require careful normalization per corpus.

By converting to ranks, RRF is **score-agnostic** — it just rewards documents that show up high in *both* lists.

### The `k` parameter intuition

| `k` | Behavior |
|-----|---------|
| **k = 0** | Top-ranked dominates: rank-1 score is 10× rank-10 |
| **k = 60** (default) | Balanced: rank-1 score is ~1.2× rank-10 |
| **k = 200** | Nearly rank-blind: rank-1 ≈ rank-10 |

So a higher `k` smooths things out — useful when both retrievers are noisy. The default of 60 (from the original RRF paper) works well in practice.

### Beta / alpha — weighted hybrid blending

Some hybrid implementations use a weighted blend instead of RRF:

$$\text{Score}(d) = \beta \cdot \text{semantic}(d) + (1 - \beta) \cdot \text{BM25}(d)$$

- `β = 0` → pure BM25
- `β = 1` → pure semantic
- `β = 0.5` → equal blend

In Weaviate this is exposed as the `alpha` parameter on `collection.query.hybrid(...)`. See [[RAG_Weaviate]].

```
Hybrid search funnel
────────────────────
                 ┌── BM25 search ─────► top 50 ┐
Query ───────────┤                              ├── Filter ──► RRF fusion ──► top K
                 └── Semantic search ─► top 50 ┘   metadata
```

---

## Additional Evaluation Metrics

Precision@K and Recall@K are covered above. Two more:

### Mean Average Precision (MAP@K)

For each **relevant** document at rank ≤ K, compute Precision@(its rank). Average across relevant docs to get **Average Precision (AP)** for one query. Average AP across all queries → **MAP**.

```
Query 1: relevant docs found at ranks 1, 3, 5 → precisions = 1.00, 0.67, 0.60
                                                AP = (1.00 + 0.67 + 0.60) / 3 = 0.76
Query 2: relevant docs found at ranks 2 only  → precisions = 0.50
                                                AP = 0.50
MAP = (0.76 + 0.50) / 2 = 0.63
```

MAP captures both **how many** relevant docs you found *and* **how well you ranked them**.

### Mean Reciprocal Rank (MRR)

For each query, find the rank of the **first** relevant document. Reciprocal rank `RR = 1 / rank`. Average across queries → MRR.

| First relevant at rank | RR |
|------------------------|-----|
| 1 | 1.0 |
| 2 | 0.5 |
| 3 | 0.33 |
| 5 | 0.2 |

```
4 queries, first-relevant ranks = [1, 3, 6, 2]
MRR = (1.0 + 0.33 + 0.17 + 0.5) / 4 = 0.50
```

MRR is the right metric when the user really only cares about the **top result** (chatbots, FAQs).

### Choosing the right metric

| Metric | Best when… |
|--------|-----------|
| **Recall@K** | "Did we find the right document anywhere in the top-K?" — the standard RAG question |
| **Precision@K** | "Is the context I'm sending to the LLM noisy?" |
| **MAP@K** | Evaluating *ranking quality* across many relevant docs |
| **MRR** | First answer matters most (FAQ chatbots, direct Q&A) |

All four require **ground truth** — labeled queries with known relevant documents.

---

## Precision/Recall in Practice — 20 Newsgroups Walkthrough

Concrete numbers from the course's evaluation lab on 20 Newsgroups (500–600 docs per category):

| K | Average Precision | Average Recall | Notes |
|---|-------------------|---------------|-------|
| 5 | ~1.00 | ~0.01 | 8/10 queries get perfect precision; recall is ~1% |
| 20 | ~0.80 | ~0.03 | "Electronics" drops 1.00 → 0.80; recall triples |
| 50 | ~0.65 | ~0.08 | "Windows OS" drops to 0.60; recall up to ~8% |

Hardest query: *"historical influence of politics on society"* — precision stuck at 0.40–0.52 across all K. Symptom of a query that overlaps multiple categories semantically.

**Takeaway:** for RAG, **K = 5–15** is the sweet spot. You don't need to find every relevant document; you need a small, clean set the LLM can read.

---

## Top-K Compensation Pattern

When comparing chunk strategies of different sizes, equalize the *total context volume* sent to the LLM by adjusting top_k:

```python
# Short chunks (~25 words each) — fetch 8 to get ~200 words of context
results_short = collection.query.near_text(query, limit=8,
                  filters=Filter.by_property("chunking_strategy").equal("fixed_size_25"))

# Long chunks (~100 words each) — fetch 2 to get ~200 words of context
results_long  = collection.query.near_text(query, limit=2,
                  filters=Filter.by_property("chunking_strategy").equal("fixed_size_100"))
```

Without this, strategies are unfairly compared.

---

## See Also

- [[RAG_Embeddings]] — how embeddings work under the hood
- [[RAG_Weaviate]] — BM25, semantic, and hybrid search via Weaviate API
- [[RAG_Prompt_Engineering]] — routing to different retrievers based on query type
- [[RAG_Advanced_Retrieval]] — query rewriting, HyDE, NER, cross-encoders, ColBERT
- [[RAG_Evaluation]] — RAGAS metrics and labeled-dataset evaluation
