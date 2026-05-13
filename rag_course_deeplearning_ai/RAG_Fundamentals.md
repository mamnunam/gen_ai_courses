# RAG Fundamentals

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a framework that grounds LLM answers in external, retrieved knowledge. Instead of relying solely on the model's parametric memory (baked in at training time), RAG dynamically fetches relevant documents at inference time and injects them into the prompt.

### Why RAG?

LLMs have two hard limitations:

- **Knowledge cutoff** — training data has a fixed end date; the model is blind to anything after it.
- **Hallucination** — models confidently generate plausible-sounding but factually wrong content.

RAG addresses both: it retrieves real, current documents and provides them as context so the LLM can ground its answer in actual evidence rather than blurry parametric memory.

---

## The RAG Loop

```
User Query
    │
    ▼
[RETRIEVE]  →  Search a document corpus  →  Return top-k relevant chunks
    │
    ▼
[AUGMENT]   →  Combine query + retrieved docs into a prompt
    │
    ▼
[GENERATE]  →  LLM produces an answer grounded in retrieved context
```

The three stages map directly to the name: **R**etrieve → **A**ugment → **G**enerate.

---

## Core Components

| Component | Role |
|-----------|------|
| **Document corpus** | The external knowledge base (news articles, FAQs, product data, etc.) |
| **Retriever** | Finds the most relevant documents for a given query |
| **Prompt template** | Combines retrieved docs + user query into an LLM-ready prompt |
| **LLM (generator)** | Produces the final answer from the augmented prompt |

---

## Minimal Python Pipeline

```python
# 1. Load corpus
corpus = [
    {"title": "GDP report", "description": "...", "url": "...", "published_at": "2024-04-25"},
    ...
]

# 2. Retrieve relevant documents (placeholder — see RAG_Retrieval_Methods for real implementations)
def retrieve(query: str, top_k: int = 5) -> list[int]:
    # Returns indices of the top-k most relevant documents
    ...

def query_corpus(indices: list[int]) -> list[dict]:
    return [corpus[i] for i in indices]

# 3. Build augmented prompt
def generate_final_prompt(query: str, top_k: int = 5, use_rag: bool = True) -> str:
    if not use_rag:
        return query

    indices = retrieve(query, top_k)
    docs = query_corpus(indices)

    formatted = "\n".join([
        f"Title: {d['title']}, Description: {d['description']}, "
        f"Published: {d['published_at']}\nURL: {d['url']}"
        for d in docs
    ])

    return (
        f"Answer the user query below. Additional context is provided.\n"
        f"Query: {query}\n"
        f"Context:\n{formatted}"
    )

# 4. Call the LLM
def llm_call(query: str, use_rag: bool = True) -> str:
    prompt = generate_final_prompt(query, use_rag=use_rag)
    response = generate_with_single_input(prompt)
    return response['content']
```

---

## RAG vs. No-RAG

| Aspect | Without RAG | With RAG |
|--------|-------------|----------|
| Knowledge scope | Training data only | External corpus + training |
| Freshness | Stale (cutoff date) | Can be up-to-date |
| Hallucination risk | Higher | Lower (grounded in retrieved text) |
| Prompt length | Short | Longer (adds retrieved docs) |
| Cost | Lower | Higher |
| Latency | Lower | Higher (retrieval step added) |

---

## Dataset Used in This Course

**BBC News Dataset** (~870 articles, April 2024):

```python
{
  'guid':         '18ba9f26...',
  'title':        'WATCH: Would you pay a tourist fee to enter Venice?',
  'description':  'From Thursday visitors making a trip...',
  'venue':        'BBC',
  'url':          'https://www.bbc.co.uk/...',
  'published_at': '2024-04-25',
  'updated_at':   '2024-04-26'
}
```

**Fashion Forward Hub** (Modules 4–5) — clothing product catalogue + FAQ list:

```python
# Product record
{
  'product_id':         '12345',
  'productDisplayName': 'Inkfruit Men\'s Little Bit More T-shirt',
  'gender':             'Men',
  'masterCategory':     'Apparel',
  'subCategory':        'Topwear',
  'articleType':        'T-shirts',
  'baseColour':         'Yellow',
  'season':             'Summer',
  'year':               '2011',
  'usage':              'Casual',
  'price':              29.99
}

# FAQ record
{
  'question': 'What is your return policy?',
  'answer':   'Items can be returned within 30 days...',
  'type':     'returns and exchanges'
}
```

---

## LLM Helper Functions

Two helpers are used throughout the course to wrap LLM calls.

### Single-turn (one prompt → one response)

```python
from utils import generate_with_single_input

response = generate_with_single_input(
    prompt="Explain RAG in one sentence.",
    role="user",          # 'user' | 'assistant' | 'system'
    temperature=0.7,
    top_p=0.9,
    max_tokens=500,
    model="Qwen/Qwen3.5-9B"
)
# Module 1–4 format:
content = response['content']

# Module 5 format (full OpenAI response):
content       = response['choices'][0]['message']['content']
total_tokens  = response['usage']['total_tokens']
prompt_tokens = response['usage']['prompt_tokens']
```

### Multi-turn (conversation with history)

```python
from utils import generate_with_multiple_input

messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "assistant", "content": "How can I help?"},
    {"role": "user",      "content": "Recommend two travel destinations."},
]
response = generate_with_multiple_input(messages, temperature=0.8)
```

### Parameter dictionary pattern

```python
# Build the call parameters as a dict, then unpack when calling
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

kwargs = generate_params_dict("Write a short poem.", temperature=1.2, top_p=0.7)
response = generate_with_single_input(**kwargs)
```

This pattern is essential for routing different query types to different LLM parameter sets.

---

## Architecture Evolution Across the Course

```
Basic (M1)
  Query → simple lookup → augmented prompt → LLM

Better retrieval (M2)
  Query → BM25 | semantic | RRF → augmented prompt → LLM

Vector DB (M3)
  Query → Weaviate (chunked corpus, hybrid search, reranking) → LLM

Routed chatbot (M4)
  Query → Router (FAQ vs Product)
               ├── FAQ → Weaviate FAQ search → LLM
               └── Product → classify (creative/technical)
                             → metadata extraction → filtered product search → LLM

Production (M5)
  Same as M4, but:
  - Simplified prompts (3× fewer tokens)
  - Optional: skip metadata LLM call → direct semantic search
  - Full OpenTelemetry tracing + Phoenix UI for cost/latency monitoring
```

---

## See Also

- [[RAG_Embeddings]] — how text becomes vectors
- [[RAG_Retrieval_Methods]] — BM25, semantic search, RRF, evaluation
- [[RAG_Chunking]] — splitting documents for embedding
- [[RAG_Weaviate]] — vector database in practice
- [[RAG_LLM_Parameters]] — controlling LLM output
- [[RAG_Prompt_Engineering]] — routing, JSON output, chatbot patterns
- [[RAG_Production]] — cost monitoring, optimization, tracing
