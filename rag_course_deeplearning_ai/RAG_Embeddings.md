# Embeddings & Similarity

## What Are Embeddings?

An **embedding** is a fixed-size numerical vector that represents a piece of text. Embedding models are trained to place semantically similar texts close together in the vector space — so "automobile" and "car" end up nearby, even though they share no letters.

In RAG, embeddings power semantic search: both the query and the documents are embedded, then similarity is measured to find the most relevant documents.

---

## Embedding Model: BAAI/bge-base-en-v1.5

The course uses this model throughout. Key facts:

| Property | Value |
|----------|-------|
| Output dimension | 768 |
| Context window | 512 tokens |
| Library | `sentence-transformers` |
| Identifier | `"BAAI/bge-base-en-v1.5"` |

```python
from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Single string → 1D vector of shape (768,)
vec = model.encode("RAG is awesome")
print(vec.shape)   # (768,)

# List of strings → 2D matrix of shape (N, 768)
vecs = model.encode(["apple", "car", "fruit", "automobile"])
print(vecs.shape)  # (4, 768)
```

---

## The 512-Token Truncation Problem

The model silently drops anything beyond its 512-token context window. The resulting embedding is **identical** to an embedding of a truncated version of the same text:

```python
import numpy as np

long_text = open("large_text.txt").read()  # e.g. 8,000 words

emb_full      = model.encode(long_text)
emb_truncated = model.encode(long_text[:3000])  # still > 512 tokens

np.array_equal(emb_full, emb_truncated)  # True — identical vectors!
```

**Implication:** Long documents must be **chunked** before embedding. See [[RAG_Chunking]].

---

## Cosine Similarity

The most common metric for comparing embeddings. It measures the **angle** between two vectors, ignoring their magnitude.

$$\text{CosSim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \cdot \|\mathbf{d}\|}$$

- Range: [−1, 1]
- **Higher = more similar** (1 = identical direction, 0 = orthogonal, −1 = opposite)
- Standard choice for semantic search

```python
import numpy as np

def cosine_similarity(v1, array_of_vectors):
    """
    Compute cosine similarity between v1 and one or many vectors.
    Returns a list of floats.
    """
    v1 = np.array(v1, dtype=np.float32).ravel()
    A  = np.atleast_2d(np.array(array_of_vectors, dtype=np.float32))

    v1_norm = np.linalg.norm(v1)
    A_norms = np.linalg.norm(A, axis=1)
    denom   = v1_norm * A_norms

    with np.errstate(divide='ignore', invalid='ignore'):
        sims = (A @ v1) / np.where(denom == 0, 1.0, denom)
    sims[denom == 0] = 0.0

    return sims.tolist()

# Single pair
v1 = model.encode("What are the primary colors")
v2 = model.encode("Yellow, red and blue")
v3 = model.encode("Cats are friendly animals")

print(cosine_similarity(v1, v2))  # [0.738] — high: semantically related
print(cosine_similarity(v1, v3))  # [0.451] — low: unrelated
```

---

## Euclidean Distance

Measures the straight-line distance between two points in the vector space.

$$\text{EucDist}(\mathbf{q}, \mathbf{d}) = \sqrt{\sum_{j=1}^{n}(q_j - d_j)^2}$$

- Range: [0, ∞)
- **Lower = more similar**

```python
def euclidean_distance(v1, array_of_vectors):
    v1 = np.array(v1, dtype=np.float32).ravel()
    A  = np.atleast_2d(np.array(array_of_vectors, dtype=np.float32))
    dists = np.sqrt(np.sum((A - v1) ** 2, axis=1))
    return dists.tolist()
```

### Cosine vs Euclidean — Key Difference

The two metrics can rank the same vectors differently because they measure different things:

```
v1 = [1, 2]
v2 = [1, 1]         ← closer in Euclidean distance to v1
array_v = [3, 2]    ← more similar in cosine (same direction as v1)
```

- Cosine: ignores vector length, only cares about direction
- Euclidean: cares about both direction and magnitude (actual distance)

**For RAG:** use cosine similarity. Embedding models normalize magnitude; direction is all that matters.

---

## Building a Semantic Retriever

```python
import joblib

# Pre-computed embeddings for the entire corpus (shape: N × 768)
EMBEDDINGS = joblib.load("embeddings.joblib")

def retrieve_relevant(query: str, documents: list[str], metric: str = "cosine") -> list[tuple]:
    """Rank documents by similarity to the query."""
    query_emb = model.encode(query)
    doc_embs  = model.encode(documents)

    if metric == "cosine":
        scores = cosine_similarity(query_emb, doc_embs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    elif metric == "euclidean":
        scores = euclidean_distance(query_emb, doc_embs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1])

    return ranked

# Example
docs = [
    "Mt. Fuji is breathtaking in autumn.",
    "Santorini offers stunning views in spring.",
    "Kyoto's cherry blossoms are beautiful.",
    "The Maldives are a summer paradise.",
]
results = retrieve_relevant("Suggest great places to visit in Asia.", docs)
# Kyoto and Mt. Fuji rank highest
```

---

## Embedding on a Corpus Index (Numpy argsort Pattern)

```python
def semantic_search(query: str, embeddings: np.ndarray, top_k: int = 5) -> list[int]:
    """Return indices of the top-k most similar documents."""
    query_emb = model.encode(query)
    scores    = cosine_similarity(query_emb, embeddings)   # list of N floats
    indices   = np.argsort(-np.array(scores))              # sort descending
    return [int(i) for i in indices[:top_k]]

top_indices = semantic_search("What are the recent news about GDP?", EMBEDDINGS)
# → [743, 673, 626, 752, 326]
```

---

## Visualizing Embeddings with PCA

Embeddings live in 768 dimensions — impossible to plot. PCA projects them to 2D for visual inspection.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

words = ['apple', 'car', 'fruit', 'automobile', 'love', 'sentiment']
embs  = model.encode(words)   # (6, 768)

pca     = PCA(n_components=2)
reduced = pca.fit_transform(embs)   # (6, 2)

plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, reduced[i], fontsize=12)
plt.title("Word Embeddings (PCA 2D)")
plt.show()

# Expected: 'apple' ↔ 'fruit' cluster; 'car' ↔ 'automobile' cluster
```

---

## How Transformer Embeddings Learn Similarity

Embeddings capture similarity based on **contexts words appear in together** during training — not just dictionary meaning. This has an important practical implication:

> "Kyoto" may score lower than "Santorini" for the query "places to visit in Asia" even though Kyoto is factually in Asia — because travel content in the training data more frequently co-occurred "Santorini" with generic travel phrases than "Kyoto".

The model learns statistical co-occurrence, not world knowledge. This is why:
- BM25 still has value for keyword-exact queries
- Hybrid search (semantic + BM25) outperforms either alone
- Domain-specific fine-tuning improves quality for specialized corpora

---

## See Also

- [[RAG_Retrieval_Methods]] — using embeddings in BM25, semantic search, and RRF
- [[RAG_Chunking]] — why chunking is needed before embedding
- [[RAG_Weaviate]] — storing and querying embeddings in a vector database
