# RAG Knowledge Base — Master Index

A complete reference for Retrieval-Augmented Generation, organized by topic. All notes are derived from the Coursera RAG course (Modules 1–5), BBC News and Fashion Forward Hub datasets.

---

## Topic Notes

### Foundations

| File | Contents |
|------|----------|
| [[RAG_Fundamentals]] | What RAG is, the R→A→G loop, LLM internals primer, why retrieval is needed, library analogy, retriever tradeoffs, 5 applications, augmented prompt template |
| [[RAG_Transformers]] | What's inside an LLM: tokenization, embeddings + position, attention (single & multi-head), feed-forward, layers, next-token prediction, autoregression, encoder vs decoder |
| [[RAG_Embeddings]] | Embedding models, contrastive training, three distance metrics (cosine, Euclidean, dot product), three embedding levels (word/sentence/document), 512-token truncation, PCA, Kyoto/Asia failure mode |

### Retrieval

| File | Contents |
|------|----------|
| [[RAG_Retrieval_Methods]] | BM25 (with TF-IDF derivation, k₁/b parameters), semantic search, RRF (with k intuition), Precision@K, Recall@K, MAP@K, MRR, K-tradeoff narrative |
| [[RAG_Chunking]] | 8 strategies: fixed-size, overlap, paragraph, mixed, recursive character, semantic, LLM-based, context-aware; Goldilocks framing; metadata attachment |
| [[RAG_Weaviate]] | All four query modes (filter/near_text/BM25/hybrid), reranking, ANN algorithms (KNN→NSW→HNSW), filter operator cheat sheet, distance metrics |
| [[RAG_Advanced_Retrieval]] | Query rewriting, NER (GLINER), HyDE, cross-encoders, ColBERT (MaxSim scoring), reranking pipelines, two-stage retrieval |

### Generation

| File | Contents |
|------|----------|
| [[RAG_LLM_Parameters]] | Temperature, top_p, top_k, repetition penalty, logit biases, peaked-vs-flat distributions, tuning workflow, multi-turn chatbot |
| [[RAG_Prompt_Engineering]] | Few-shot routers, JSON extraction, FAQ/product pipeline, progressive filter relaxation, messages format + system prompts, chat templates, scratchpad/CoT, reasoning models, context pruning |

### Reliability & Production

| File | Contents |
|------|----------|
| [[RAG_Hallucinations]] | Hallucination types, grounding prompts, citation generation, self-consistency, ContextCite, ALCE benchmark |
| [[RAG_Evaluation]] | RAGAS (Response Relevancy, Faithfulness), evaluator scope matrix, A/B testing, LLM selection benchmarks (MMLU, LLM Arena, ELO), benchmark saturation |
| [[RAG_Agentic_RAG]] | Multi-LLM workflows (sequential/conditional/iterative/parallel), RAG vs fine-tuning, cheap-router/expensive-answer pattern |
| [[RAG_Production]] | Token cost baseline, prompt optimization (65% reduction), OpenTelemetry + Phoenix tracing, evaluator scope, quantization (1-bit, Matryoshka), caching, security, multi-modal RAG |

---

## Architecture Evolution (Modules 1 → 5)

```
M1 — Foundations
  Minimal RAG loop: embed query → cosine similarity → inject top-k → LLM call

M2 — Retrieval Methods
  + BM25 keyword search (bm25s)
  + Semantic search (dense retrieval)
  + RRF hybrid fusion
  + Precision@K / Recall@K evaluation

M3 — Vector Database (Weaviate)
  + Persistent vector index (HNSW)
  + Metadata filtering + near_text + BM25 + hybrid (alpha)
  + Cross-encoder reranking
  + Chunking strategies → multi-strategy comparison

M4 — Production Chatbot
  + FAQ vs Product router (few-shot classification)
  + Creative vs Technical sub-router
  + LLM metadata extraction → Weaviate filters
  + Progressive filter relaxation
  + Multi-turn conversation context
  + Phoenix tracing (basic)

M5 — Optimization & Observability
  + Full token cost baseline (~2,650 tokens/query)
  + Simplified prompts → ~940 tokens/query (−65%)
  + `simplified` boolean flag pattern
  + OpenTelemetry spans, traces, nested spans
  + Auto-instrumentation for OpenAI-compatible calls
  + Phoenix cost tracking UI
```

---

## Glossary

**Agentic RAG** — Multi-LLM workflow where each LLM does one specialized task (routing, evaluation, generation, citation). Patterns: sequential, conditional, iterative, parallel.

**ALCE** — Automatic LLMs' Citation Evaluation benchmark. Tests citation quality, correctness, and fluency.

**Alpha** — Weaviate hybrid search blend parameter. `0.0` = pure BM25, `1.0` = pure vector, `0.5` = equal blend.

**Attention** — Mechanism in a transformer where every token weights every other token's contribution to its meaning. The conceptual heart of the architecture.

**Auto-instrument** — `register(auto_instrument=True)` in Phoenix/OpenTelemetry automatically traces all `openai.chat.completions.create()` calls without extra span code.

**Autoregressive** — Each generated token is appended to the input, and the next token is predicted given the full updated sequence.

**Bi-encoder** — Default semantic retriever architecture: query and document are embedded separately, then compared by cosine similarity. Fast but less precise than cross-encoders.

**BM25** — Best Match 25. Sparse keyword-based retrieval. Scores documents by term frequency + inverse document frequency, with two refinements (TF saturation via `k₁`, length normalization via `b`). Strong on exact matches and rare terms.

**Chain** — OpenInference span kind for multi-step orchestration functions (e.g., `answer_query`).

**Chain-of-Thought (CoT)** — Prompting pattern triggering step-by-step reasoning ("Let's think step by step"). Improves quality on multi-step problems.

**Chunking** — Splitting documents into smaller pieces before embedding. Required when text exceeds the embedding model's token limit (512 for `bge-base-en-v1.5`).

**ColBERT** — Contextualized Late Interaction Over BERT. One vector per token; scores via MaxSim (each query token finds its closest doc token; sum maxes). Near-cross-encoder quality at decent speed; high storage cost.

**ContextCite** — Tool that attributes each generated sentence back to retrieved documents. Useful for citation generation and evaluation.

**Context-aware chunking** — LLM-augmented chunking that adds a brief context label to each chunk before embedding. Highest-value chunking improvement.

**Contrastive training** — How embedding models learn: push positive pairs together, pull negative pairs apart in vector space.

**Cosine similarity** — Measures the angle between two vectors: `(A·B) / (‖A‖·‖B‖)`. Range: −1 to 1. Used for embedding comparison; ignores magnitude.

**Cross-encoder** — Reranking architecture: query + document are concatenated and run through the model together. Much higher quality than bi-encoders; doesn't scale beyond ~100 candidates.

**Dense retrieval** — Synonym for semantic search. Queries and documents are both embedded; similarity measured by dot product or cosine distance.

**Dot product** — Third distance metric beyond cosine and Euclidean. Un-normalized form of cosine similarity.

**Embedding** — A fixed-length vector (e.g., 768 dimensions) that encodes semantic meaning. Similar texts → nearby vectors.

**Faithfulness** — RAGAS metric measuring whether response claims are supported by retrieved documents. Primary anti-hallucination metric.

**Few-shot classification** — Prompt pattern that gives the LLM labeled examples + category definitions and asks it to classify a new input in one or two tokens (`temperature=0`, `max_tokens=10`).

**Fine-tuning** — Adjusting model weights via Supervised Fine-Tuning (SFT) on labeled examples. Teaches *style* and *task patterns* well; teaches new facts poorly. Compare to RAG (knowledge injection).

**GLINER** — Generalist and Lightweight NER model used for query parsing (extracting entities from user queries).

**Greedy decoding** — Default generation: always pick the highest-probability next token. Deterministic but can fall into repetition loops.

**HNSW** — Hierarchical Navigable Small World. Approximate nearest-neighbor index used by Weaviate. Stacks NSW graphs at decreasing sparsity for logarithmic scaling.

**HyDE** — Hypothetical Document Embeddings. Generate a hypothetical answer with an LLM, embed *that*, search with the hypothetical-answer vector. Closes the query-doc domain mismatch.

**Hybrid search** — Combining sparse (BM25) and dense (vector) retrieval, typically via RRF or a weighted blend.

**LLM (Large Language Model)** — The "Generate" step of RAG. Takes the augmented prompt (query + retrieved context) and produces the final answer.

**LLM Arena** — chat.lmsys.org. Pairwise human evaluation of LLMs producing an ELO-style leaderboard.

**LLM-as-judge** — Evaluator pattern where an LLM scores outputs of another LLM. Cheap, flexible, biased toward own model family.

**Logit bias** — Direct per-token override of LLM probabilities. Used to ban tokens or boost classifier labels.

**MAP@K** — Mean Average Precision at K. Captures both how many relevant docs were found *and* ranking quality.

**Matryoshka quantization** — Vector dimensions sorted by information density at training time; use first N dims for fast retrieval, full vector for rerank.

**MMLU** — Massive Multitask Language Understanding benchmark. 57 subjects, multiple choice.

**MRR** — Mean Reciprocal Rank. `1 / (rank of first relevant doc)`, averaged across queries. Right metric when only the top result matters.

**NER** — Named Entity Recognition. Extracting people/places/dates/organizations from queries for use as metadata filters.

**Nucleus sampling** — See *Top-p*.

**OpenInference** — A tracing specification for LLM applications, used by Phoenix to categorize spans (chain, tool, llm, retriever).

**OpenTelemetry** — Open-source observability framework for distributed systems. Phoenix uses it to collect RAG pipeline traces.

**Overlap** — In chunking, repeating the tail of the previous chunk at the start of the next. Prevents information loss at boundaries. Typical: 15–25% of chunk size.

**Phoenix** — Arize AI's open-source observability tool for LLM apps. Runs locally at `http://localhost:6006`. Displays traces, token counts, and costs.

**Precision@K** — Of the top-K retrieved documents, what fraction are relevant? `|relevant ∩ retrieved| / K`.

**Progressive filter relaxation** — When a filtered Weaviate query returns too few results, iteratively drop filters from least to most important until a sufficient result set is found.

**Quantization** — Compressing float embeddings (or LLM weights) to lower-bit integers. 4-step process: find min/max → divide into N buckets → assign ints → store min + scale.

**Query rewriting** — Using an LLM to clarify/expand an ambiguous user query before retrieval.

**RAGAS** — Python library for RAG evaluation. Key metrics: Response Relevancy, Faithfulness, Context Precision, Context Recall, Noise Sensitivity.

**Reasoning model** — LLM that internalizes Chain-of-Thought during training. Emits "reasoning tokens" before the visible answer. Higher accuracy but slower and pricier.

**Response Relevancy** — RAGAS metric: does the response actually answer the question? Mechanism: generate likely-prompts from the response, compare to actual prompt.

**RAG (Retrieval-Augmented Generation)** — Augmenting an LLM's answer with documents retrieved at query time, rather than relying solely on parametric memory. Reduces hallucinations; enables access to up-to-date or private data.

**Recall@K** — Of all relevant documents, what fraction appear in the top-K retrieved? `|relevant ∩ retrieved| / |relevant|`.

**Recency bias** — LLMs give more weight to text that appears later in the prompt. Exploit this by placing the most relevant retrieved document nearest the query.

**Reciprocal Rank Fusion (RRF)** — Hybrid ranking formula: `Score(d) = Σ 1 / (k + rank_r(d))`, where `k=60`. Merges ranked lists from multiple retrievers without requiring score normalization. Score-agnostic.

**Reranking** — A second-pass cross-encoder model that re-scores retrieved results using full attention over (query, document) pairs. More accurate than bi-encoder retrieval but slower.

**Retriever** — OpenInference span kind for vector DB queries.

**Scratchpad** — Prompt pattern asking the LLM to reason inside `<scratchpad>...</scratchpad>` tags before producing the final answer.

**Semantic chunking** — Chunking strategy using cosine similarity between consecutive sentences to detect topic boundaries.

**Semantic search** — Retrieval based on meaning similarity, using embedding vectors and cosine distance. Handles paraphrase and synonymy well; weaker on rare exact-match terms.

**Simplified flag** — Boolean `simplified=True` parameter threaded through every pipeline function. Enables A/B testing, gradual rollout, and easy rollback of optimized prompts.

**Sparse retrieval** — Retrieval based on term overlap (BM25, TF-IDF). Vectors are sparse (mostly zeros); each dimension corresponds to a vocabulary term.

**Span** — A single timed operation in a trace: name, start/end time, attributes (inputs, outputs, token counts). Atomic unit of observability.

**Temperature** — Reshapes the LLM's next-token probability distribution. `0` = deterministic (argmax). `~1.0` = natural. `>1.5` = increasingly random.

**Tool** — OpenInference span kind for helper functions (routing, formatting).

**Top-k** — Restricts LLM sampling to the k most probable tokens at each step.

**Top-p (nucleus sampling)** — Restricts LLM sampling to the smallest set of tokens whose cumulative probability ≥ p. Adapts to distribution shape; generally preferred over top-k.

**Trace** — A tree of spans representing one complete end-to-end request through the RAG system.

**Vectorizer** — Weaviate component that automatically embeds object properties when data is ingested. Configured per-collection (e.g., `text2vec-transformers`).

---

## Quick Reference: Retrieval Methods

| Method | Library / API | Strengths | Weaknesses |
|--------|--------------|-----------|------------|
| BM25 | `bm25s` / Weaviate `.bm25()` | Exact term match, rare words, interpretable | Misses synonyms, paraphrase |
| Semantic | `sentence-transformers` / Weaviate `.near_text()` | Meaning, paraphrase, multilingual | Can miss rare exact terms |
| Hybrid (RRF) | Custom / Weaviate `.hybrid()` | Best of both; robust default | Adds latency |
| Filtered | Weaviate `Filter.all_of(...)` | Precise scoping (gender, price, season) | Returns nothing if over-filtered |
| Reranking | Weaviate `Rerank(...)` | Improves ordering of noisy results | Expensive; needs a cross-encoder model |

---

## Quick Reference: LLM Parameters by Task

| Task | Temperature | Top-p | Max Tokens |
|------|-------------|-------|------------|
| Classification / routing | 0 | 0.1 | 1–10 |
| JSON / structured output | 0–0.3 | 0.7 | 500–1500 |
| Technical product queries | 0.3 | 0.7 | 500 |
| FAQ answers | 0.5 | 0.8 | 500 |
| General conversation | 0.7 | 0.9 | 500 |
| Creative outfit suggestions | 1.0 | 0.9 | 500 |
| Creative writing / poems | 1.2 | 0.8 | 300 |

---

## Quick Reference: Chunking Strategy

| Query Type | Recommended Strategy | Chunk Size |
|------------|---------------------|------------|
| Specific fact / command | Fixed-size | 25–50 words |
| General question | Fixed-size with overlap | 100 words, 20% overlap |
| Document with structure | Mixed (variable + min_words merge) | Section-based |
| Short document (<512 tokens) | No chunking needed | — |
| Code / step-by-step instructions | Fixed-size with overlap | 100–150 words, 25% overlap |

---

## Quick Reference: Token Costs (Fashion Forward Hub)

| Step | Full Prompt | Simplified | Saving |
|------|-------------|------------|--------|
| FAQ vs Product router | ~200 tokens | ~80 tokens | −60% |
| Creative vs Technical classifier | ~150 tokens | ~60 tokens | −60% |
| FAQ lookup | ~1,200 tokens | ~350 tokens | −71% |
| Metadata extraction | ~1,500 tokens | 0 tokens | −100% |
| Final answer generation | ~800 tokens | ~800 tokens | 0% |
| **Total (product query)** | **~2,650** | **~940** | **−65%** |

---

## Key Code Patterns

### RRF Formula

```python
def reciprocal_rank_fusion(results: list[list], k: int = 60) -> list[tuple]:
    scores = {}
    for result_list in results:
        for rank, doc in enumerate(result_list):
            scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Cosine Similarity

```python
import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
```

### Weaviate Hybrid Search

```python
response = collection.query.hybrid(
    query="cheap winter travel",
    filters=Filter.by_property("budget").contains_any(["Low", "Moderate"]),
    alpha=0.5,   # 0=BM25, 1=vector, 0.5=blend
    limit=5
)
```

### Few-Shot Router

```python
PROMPT = f"""Classify for a clothing store.
FAQ: policies/shipping/contact/returns. Product: items/prices/outfits/styles.
Examples: "return policy?" FAQ | "blue dress?" Product | "store hours?" FAQ | "sunny look?" Product
One word only: FAQ or Product.
Query: {query}"""

kwargs   = generate_params_dict(PROMPT, temperature=0, max_tokens=10)
response = generate_with_single_input(**kwargs)
```

### Phoenix Span Template

```python
with tracer.start_as_current_span("step_name", openinference_span_kind="tool") as span:
    span.set_input({"key": value})
    try:
        result = do_work()
        span.set_output(result)
        span.set_status(Status(StatusCode.OK))
    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR))
        raise
```

### Parameter Dict Pattern

```python
def generate_params_dict(
    prompt: str,
    temperature: float = 1.0,
    role: str = "user",
    top_p: float = 1.0,
    max_tokens: int = 500,
    model: str = "Qwen/Qwen3.5-9B"
) -> dict:
    return {
        "prompt": prompt, "role": role,
        "temperature": temperature, "top_p": top_p,
        "max_tokens": max_tokens, "model": model
    }
```

---

## Libraries Used

| Library | Purpose |
|---------|---------|
| `sentence-transformers` | Load embedding model (`bge-base-en-v1.5`), embed text |
| `bm25s` | Fast BM25 implementation (indexing + retrieval) |
| `weaviate-client` | Vector DB client (all query modes, collection management) |
| `openai` | OpenAI-compatible LLM client (used with Together.ai base URL) |
| `phoenix` (`arize-phoenix`) | Local observability UI for traces and cost tracking |
| `opentelemetry` | Tracing SDK; Phoenix uses `phoenix.otel.register()` as the tracer provider |
| `pydantic` | Structured output schema definition for JSON-format LLM responses |
| `numpy` | Cosine similarity, vector arithmetic, `argsort` for retrieval |
| `sklearn` | `TfidfVectorizer`, `PCA` for visualization |
| `tqdm` | Progress bars for Weaviate batch insert |

---

## See All Notes

### Foundations
- [[RAG_Fundamentals]]
- [[RAG_Transformers]]
- [[RAG_Embeddings]]

### Retrieval
- [[RAG_Retrieval_Methods]]
- [[RAG_Chunking]]
- [[RAG_Weaviate]]
- [[RAG_Advanced_Retrieval]]

### Generation
- [[RAG_LLM_Parameters]]
- [[RAG_Prompt_Engineering]]

### Reliability & Production
- [[RAG_Hallucinations]]
- [[RAG_Evaluation]]
- [[RAG_Agentic_RAG]]
- [[RAG_Production]]
