# RAG Knowledge Base
 
A complete topic-by-topic reference for **Retrieval-Augmented Generation**, distilled from DeepLearning.AI / Coursera's *Retrieval-Augmented Generation (RAG)* course (Modules 1–5) plus the accompanying notebooks and the BBC News and Fashion Forward Hub datasets used in the labs.
 
The notes are designed to work as a wiki — every `[[Topic]]` reference resolves to a file in this repo. Start at **[RAG_Index.md](./RAG_Index.md)**.
 
---
 
## Quick start
 
| If you want to… | Read |
|------|------|
| Understand what RAG is | [RAG_Fundamentals.md](./RAG_Fundamentals.md) |
| Browse all topics with quick-reference tables and a glossary | [RAG_Index.md](./RAG_Index.md) |
| Build your first retrieval pipeline | [RAG_Retrieval_Methods.md](./RAG_Retrieval_Methods.md) |
| Wire up a vector database | [RAG_Weaviate.md](./RAG_Weaviate.md) |
| Build a production chatbot | [RAG_Prompt_Engineering.md](./RAG_Prompt_Engineering.md) → [RAG_Production.md](./RAG_Production.md) |
 
---
 
## Contents
 
### Foundations
 
- **[RAG_Fundamentals.md](./RAG_Fundamentals.md)** — What RAG is, the Retrieve → Augment → Generate loop, why LLMs need retrieval, the library analogy, retriever tradeoffs, five applications, and the canonical augmented-prompt template.
- **[RAG_Transformers.md](./RAG_Transformers.md)** — What's inside an LLM: tokenization, position + semantic embeddings, single- and multi-head attention, feed-forward layers, next-token prediction, and the autoregressive loop.
- **[RAG_Embeddings.md](./RAG_Embeddings.md)** — Embedding models (`BAAI/bge-base-en-v1.5`), contrastive training, cosine vs. Euclidean vs. dot product, the 512-token truncation gotcha, PCA visualization, and the "embeddings learn co-occurrence, not truth" failure mode.
### Retrieval
 
- **[RAG_Retrieval_Methods.md](./RAG_Retrieval_Methods.md)** — BM25 (with TF-IDF derivation and the `k₁` / `b` parameters), semantic search from scratch with `sentence-transformers`, Reciprocal Rank Fusion (with the role of `k`), and four evaluation metrics: Precision@K, Recall@K, MAP@K, MRR.
- **[RAG_Chunking.md](./RAG_Chunking.md)** — Eight chunking strategies (fixed-size, overlap, paragraph, mixed, recursive character, semantic, LLM-based, context-aware), the Goldilocks framing, and a decision flowchart for picking one.
- **[RAG_Weaviate.md](./RAG_Weaviate.md)** — Collections, batch ingest with `generate_uuid5`, all four query modes (metadata filter, `near_text`, BM25, hybrid with `alpha`), cross-encoder reranking, the KNN → NSW → HNSW algorithmic progression, and a filter-operator cheat sheet.
- **[RAG_Advanced_Retrieval.md](./RAG_Advanced_Retrieval.md)** — Query rewriting, NER-based query parsing (GLINER), HyDE (Hypothetical Document Embeddings), bi-encoder vs. cross-encoder vs. ColBERT (with worked MaxSim score), and the two-stage retrieve-then-rerank pattern.
### Generation
 
- **[RAG_LLM_Parameters.md](./RAG_LLM_Parameters.md)** — Temperature, top-p, top-k, repetition penalty, logit biases, the peaked-vs-flat distribution intuition, a recommended tuning workflow, and a parameter-recommendations-by-task table.
- **[RAG_Prompt_Engineering.md](./RAG_Prompt_Engineering.md)** — Few-shot classification routers, structured JSON output (prompt-based and Pydantic), the FAQ-vs-Product chatbot architecture, progressive filter relaxation, the messages format with system prompts, chat-template internals, in-context-learning vocabulary, `<scratchpad>` / Chain-of-Thought / reasoning models, and context-pruning strategies.
### Reliability & Production
 
- **[RAG_Hallucinations.md](./RAG_Hallucinations.md)** — Types of hallucination, why they happen, grounding prompts, citation generation, self-consistency methods, ContextCite, and the ALCE benchmark.
- **[RAG_Evaluation.md](./RAG_Evaluation.md)** — RAGAS metrics (Response Relevancy, Faithfulness, Context Precision/Recall), the evaluator-scope matrix (component vs. system × code-based / LLM-as-judge / human), A/B testing, and LLM-selection benchmarks (MMLU, LLM Arena ELO).
- **[RAG_Agentic_RAG.md](./RAG_Agentic_RAG.md)** — Multi-LLM workflows (sequential, conditional, iterative, parallel), RAG vs. fine-tuning (with the decision rule "RAG = knowledge injection; fine-tuning = domain adaptation"), and the cheap-router / expensive-answer cost pattern.
- **[RAG_Production.md](./RAG_Production.md)** — Token cost baseline (~2,650 tokens/query), a 65% prompt-optimization walkthrough, the `simplified` flag pattern, OpenTelemetry + Phoenix tracing (decorators, manual spans, auto-instrument), quantization (4-step, 1-bit, Matryoshka), caching, component-level latency budgets, security (vector reconstruction attacks), and multi-modal RAG.
### Master index
 
- **[RAG_Index.md](./RAG_Index.md)** — Topic table, architecture-evolution diagram (M1 → M5), a full glossary, and quick-reference tables for retrieval methods, LLM parameters by task, chunking strategies, and token costs.
---
 
## How the architecture builds, module by module
 
```
M1 — Foundations
  Minimal RAG loop: embed query → cosine similarity → inject top-k → LLM call
 
M2 — Retrieval Methods
  + BM25, semantic, RRF hybrid; Precision@K / Recall@K evaluation
 
M3 — Vector Database (Weaviate)
  + Persistent HNSW index, metadata filtering, hybrid (alpha), cross-encoder reranking
  + Chunking strategies
 
M4 — Production Chatbot
  + Few-shot routers, sub-routers, LLM metadata extraction, filter relaxation
  + Multi-turn conversation, basic Phoenix tracing
 
M5 — Optimization & Observability
  + Simplified prompts (~65% token reduction), OpenTelemetry spans + traces
  + Phoenix cost tracking, full evaluator suite
```
 
---
 
## Source materials
 
The notes were built by reading every cell of the course notebooks (`C1M2_Ungraded_Lab_*`, `C1M3_*`, `C1M4_*`, `C1M5_*` — both ungraded labs and graded assignments) and every page of the five module slide decks (`RAG_M1.pdf` … `RAG_M5.pdf`). Cross-references and code examples are pulled from those sources.
 
## Tools and libraries covered
 
`sentence-transformers` · `bm25s` · `weaviate-client` · `openai` / Together.ai · `pydantic` · `phoenix` (`arize-phoenix`) · `opentelemetry` · `numpy` · `sklearn` · `tqdm` · `gliner` (NER) · `RAGAS` (evaluation)
 
## Recommended reading order
 
For a first pass:
 
1. [RAG_Fundamentals.md](./RAG_Fundamentals.md)
2. [RAG_Transformers.md](./RAG_Transformers.md) *(optional but clarifies the rest)*
3. [RAG_Embeddings.md](./RAG_Embeddings.md)
4. [RAG_Retrieval_Methods.md](./RAG_Retrieval_Methods.md)
5. [RAG_Chunking.md](./RAG_Chunking.md)
6. [RAG_Weaviate.md](./RAG_Weaviate.md)
7. [RAG_LLM_Parameters.md](./RAG_LLM_Parameters.md)
8. [RAG_Prompt_Engineering.md](./RAG_Prompt_Engineering.md)
9. [RAG_Production.md](./RAG_Production.md)
Then dig into the four reliability/advanced notes as needed:
 
- [RAG_Hallucinations.md](./RAG_Hallucinations.md)
- [RAG_Evaluation.md](./RAG_Evaluation.md)
- [RAG_Advanced_Retrieval.md](./RAG_Advanced_Retrieval.md)
- [RAG_Agentic_RAG.md](./RAG_Agentic_RAG.md)
---
 
## License & attribution
 
These are personal study notes derived from the DeepLearning.AI / Coursera *Retrieval-Augmented Generation* course taught by Zain Hasan. Concepts and code patterns belong to the course authors; the synthesis and prose here are mine. Use the notes as study material, not as a substitute for the course itself.
