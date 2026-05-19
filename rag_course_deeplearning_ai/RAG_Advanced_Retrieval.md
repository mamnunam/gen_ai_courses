# Advanced Retrieval

Techniques beyond plain BM25 / semantic / hybrid: query rewriting, NER-based query parsing, HyDE, cross-encoders, ColBERT, and reranking pipelines.

---

## Query Processing — Make the Query Better Before You Search

A poorly-worded query produces poorly-relevant results. Three techniques fix this upstream of retrieval.

### 1. Query Rewriting

Use an LLM to rewrite ambiguous, conversational user queries into a clearer, retrieval-friendly form.

```
User query:
  "Yesterday I felt a hard pull on the leash when my dog ran. Now my arm
   has been numb for three days. What's wrong with me?"

Rewritten query (via LLM):
  "Experienced a sudden, forceful pull on the shoulder, resulting in
   persistent shoulder numbness and finger numbness for three days.
   What are the potential causes or diagnoses, such as neuropathy or
   nerve impingement?"
```

The rewriter prompt typically instructs:
- Clarify ambiguous phrases
- Use domain-specific terminology (medical/legal/technical)
- Add synonyms for keyword retrieval
- Remove unnecessary or distracting information

```python
REWRITE_PROMPT = """Rewrite the following user query for retrieval over a medical knowledge base.
- Clarify ambiguous phrases
- Use medical terminology where applicable
- Add synonyms
- Remove distracting information

User query: {query}
Rewritten query:"""

def rewrite_query(query: str) -> str:
    response = generate_with_single_input(
        REWRITE_PROMPT.format(query=query),
        temperature=0.3, max_tokens=300
    )
    return response["choices"][0]["message"]["content"].strip()
```

**Tradeoff:** adds one LLM call (~200–300 ms, 200 tokens). Worth it for ambiguous user input; skip for clean structured queries.

### 2. Named Entity Recognition (NER)

Extract entities — people, places, dates, organizations — from the query and use them as **metadata filters**, not just text input.

```
"I read The Great Gatsby by F. Scott Fitzgerald last summer while visiting
 New York in the 1920s..."
                                                  ↓
[BOOK: The Great Gatsby]
[PERSON: F. Scott Fitzgerald]
[LOCATION: New York]
[DATE: 1920s]
```

**Library to know:** **GLINER** (Generalist and Lightweight NER) — a small NER model that works zero-shot across many entity types.

Pipeline:

```python
from gliner import GLiNER

ner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
ENTITY_TYPES = ["person", "location", "organization", "date", "book", "product"]

def parse_query(query: str) -> dict:
    entities = ner_model.predict_entities(query, ENTITY_TYPES)
    parsed = {"text": query, "entities": {}}
    for ent in entities:
        parsed["entities"].setdefault(ent["label"], []).append(ent["text"])
    return parsed

# Use in retrieval: combine semantic search with NER-derived filters
parsed = parse_query("Find me Edgar Allan Poe poems written in 1845")
filters = []
if "person" in parsed["entities"]:
    filters.append(Filter.by_property("author").contains_any(parsed["entities"]["person"]))
if "date" in parsed["entities"]:
    filters.append(Filter.by_property("year").contains_any(parsed["entities"]["date"]))

results = collection.query.near_text(query, filters=Filter.all_of(filters), limit=10)
```

**When to use:** queries where specific entities matter (legal cases by judge name, papers by author, products by brand).

### 3. HyDE — Hypothetical Document Embeddings

Counterintuitive trick: ask the LLM to write a **hypothetical answer** to the query, embed that hypothetical document, and use *its* vector for retrieval.

```
User query: "How do I treat shoulder neuropathy from a leash pull?"
                          ↓
LLM generates a hypothetical answer:
  "Shoulder neuropathy from sudden traction injuries typically presents
   with numbness, tingling, and weakness. Treatment includes rest, ice,
   NSAIDs, physical therapy, and in persistent cases, EMG evaluation
   to rule out nerve impingement..."
                          ↓
Embed the hypothetical answer (not the original query)
                          ↓
Search the corpus with the hypothetical-answer vector
```

**Why this works:** the corpus contains *documents* (answers), so embedding a query-shaped string and comparing to document-shaped strings is a domain shift. HyDE removes the shift by matching documents to documents.

```python
def hyde_retrieve(query: str, collection, top_k: int = 5):
    hypothetical = generate_with_single_input(
        f"Write a detailed paragraph answering this question:\n{query}",
        temperature=0.3, max_tokens=300
    )["choices"][0]["message"]["content"]
    
    return collection.query.near_text(hypothetical, limit=top_k)
```

**Tradeoff:** adds one LLM call. Best on ambiguous queries; little benefit on clean keyword queries.

---

## Cross-Encoders vs Bi-Encoders

### Bi-encoder (the default)

Query and document are embedded **separately**, then compared by cosine similarity:

```
Query   ──► [embed]   →  q_vec  ─┐
                                  ├──► cosine_similarity → score
Document ──► [embed]  →  d_vec  ─┘
```

- **Pros:** document vectors pre-computed once; queries are cheap (one forward pass + ANN lookup)
- **Cons:** query and doc never "see" each other inside the model → misses subtle relevance signals
- **Use case:** the standard semantic-search retriever

### Cross-encoder

Query and document are **concatenated** and run through the model together:

```
[CLS] query [SEP] document [SEP]  ──► transformer ──► single relevance score
```

Concrete example for *"Great places to eat in New York"*:

| Document | Cross-encoder score |
|----------|---------------------|
| "Best restaurants in Manhattan, top picks for 2024" | 0.85 |
| "The cuisine of New York City comprises many..." | 0.70 |
| "NYC subway system map and tips" | 0.12 |

- **Pros:** much higher quality — the model sees query and document jointly with full attention.
- **Cons:** can't pre-compute; every (query, doc) pair needs its own forward pass. **Scales terribly to millions of documents.**
- **Use case:** **reranking** a small set (10–100 candidates) from a fast initial retriever.

### ColBERT — the middle ground

ColBERT (**Contextualized Late Interaction Over BERT**) computes one vector **per token** for both query and document:

```
Query    ──► [tok_1_vec, tok_2_vec, ..., tok_M_vec]
Document ──► [tok_1_vec, tok_2_vec, ..., tok_N_vec]
```

Scoring uses **MaxSim**: for each query token, find the most-similar document token; sum the maxes.

Worked example — query *"Great places to eat in New York"* against doc *"The cuisine of New York City comprises many..."*:

```
Query token   →   max similarity over doc tokens
─────────────────────────────────────────
"Great"       →   0.3
"places"      →   0.3
"to"          →   0.2
"eat"         →   0.7  (matches "cuisine")
"in"          →   0.3
"New"         →   0.9  (matches "New")
"York"        →   0.9  (matches "York")
                  ─────
                  Σ = 3.6  ← ColBERT score
```

| Method | Quality | Speed | Storage |
|--------|---------|-------|---------|
| Bi-encoder | Baseline | Fast | Minimal (1 vec/doc) |
| Cross-encoder | Best | Slow | Minimal |
| **ColBERT** | Near-cross-encoder | Decent | **High** (N vecs/doc) |

---

## Reranking — the Two-Stage Retrieval Pattern

Almost every production RAG uses this two-stage pattern:

```
Stage 1: Fast, imprecise initial retrieval
   bi-encoder semantic + BM25 + filters     →  top 20-50 candidates

Stage 2: Slow, precise reranking
   cross-encoder OR LLM scoring             →  top 3-5 best
```

### The "Capital of Canada" demo

Initial retrieval returns these top candidates (bi-encoder lexical overlap, but ranked badly):

```
1. "Toronto is Canada's largest city..."
2. "Paris is the capital of France..."
3. "Canada is the maple syrup capital of the world..."
4. "Ottawa is the capital of Canada..."     ← buried at rank 4
```

After cross-encoder reranking on the query *"What is the capital of Canada?"*:

```
1. "Ottawa is the capital of Canada..."     ← promoted to rank 1
2. "Toronto is Canada's largest city..."
3. "Canada is the maple syrup capital..."
4. "Paris is the capital of France..."
```

### LLM-based reranking

An alternative to cross-encoders: prompt a small fine-tuned LLM to score relevance directly.

```python
RERANK_PROMPT = """Score the relevance of this document to the query, 0.00 (irrelevant) to 1.00 (perfect).
Output a single number, nothing else.

Query: {query}
Document: {doc}
Score:"""

def llm_rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    scored = []
    for doc in candidates:
        response = generate_with_single_input(
            RERANK_PROMPT.format(query=query, doc=doc["chunk"]),
            temperature=0, max_tokens=5
        )
        score = float(response["choices"][0]["message"]["content"].strip())
        scored.append((score, doc))
    return [d for _, d in sorted(scored, reverse=True)[:top_k]]
```

Powerful but **costly**: one LLM call per candidate. Reserve for highest-value reranking layers.

---

## Putting It All Together — A Full Retrieval Stack

A production-grade pipeline often layers several techniques:

```
User Query
    │
    ▼
[Query Rewriting]  (LLM, ~250 ms)
    │
    ▼
[NER → metadata filters]
    │
    ▼
[Hybrid Search (BM25 + semantic)]  → top 50
    │
    ▼
[Cross-Encoder Reranker]            → top 5
    │
    ▼
[Augmented Prompt]                  → LLM
```

Add components only when each one demonstrably improves your eval metrics. Latency and cost add up:

| Stage | Typical latency | Quality contribution |
|-------|----------------|---------------------|
| Query rewriting | 200–300 ms | High for ambiguous queries |
| NER parsing | < 50 ms | High when entities matter |
| Hybrid retrieval | < 20 ms | Baseline quality |
| Cross-encoder rerank | 50 ms (for 20 docs) | Often dramatic |
| LLM rerank | 1–2 s | Marginal over cross-encoder |

---

## See Also

- [[RAG_Retrieval_Methods]] — the BM25 / semantic / RRF foundations
- [[RAG_Weaviate]] — Weaviate's built-in reranker via `Rerank(prop=..., query=...)`
- [[RAG_Embeddings]] — bi-encoder embedding fundamentals
- [[RAG_Prompt_Engineering]] — the prompts that drive query rewriters and rerankers
- [[RAG_Production]] — latency budgets across these stages
