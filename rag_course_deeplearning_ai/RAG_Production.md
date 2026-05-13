# Production RAG — Cost, Optimization & Observability

## The Production Problem

A RAG pipeline that works in a notebook often becomes expensive and slow at scale. Two failure modes:

1. **Rising costs** — every query triggers multiple LLM calls; costs scale with traffic with no visibility into where money goes.
2. **High latency** — long prompts and sequential LLM calls add up; users experience slow responses.

The solutions are prompt optimization (do less per query) and observability (see exactly what's happening).

---

## 1. Token Cost Baseline

In the full Fashion Forward Hub pipeline from Module 4, each product query triggers:

| Step | LLM call | Approx. tokens |
|------|----------|---------------|
| FAQ vs Product router | Yes | ~200 |
| Creative vs Technical classifier | Yes | ~150 |
| Metadata JSON extraction | Yes | **~1,500** |
| Final answer generation | Yes | ~800 |
| **Total per query** | | **~2,650** |

The metadata extraction step alone accounts for over half the token usage.

### Reading token counts

```python
result = generate_with_single_input("What are the primary colors?")

# Module 5 response format (full OpenAI-style)
content           = result['choices'][0]['message']['content']
prompt_tokens     = result['usage']['prompt_tokens']
completion_tokens = result['usage']['completion_tokens']
total_tokens      = result['usage']['total_tokens']
model_name        = result['model']
```

---

## 2. Prompt Optimization

### Principle: same accuracy, fewer tokens

Test simplified prompts against the full ones on a labeled query set before deploying. Accuracy must hold; only then does the token saving count.

### Router: ~200 → ~80 tokens

```python
# Full (200 tokens)
PROMPT_FULL = f"""Label the following instruction as FAQ or Product for a clothing store.
Product: specific product info or outfit creation.
FAQ: policies, contact, store info.
Examples:
    Is there a refund? Label: FAQ
    Where are stores? Label: FAQ
    Cheapest T-shirts? Label: Product
    ... (8 examples)
Return only: FAQ or Product.
Query: {query}"""

# Simplified (80 tokens) — identical accuracy on test set
PROMPT_SIMPLIFIED = f"""Classify for a clothing store.
FAQ: policies/shipping/contact. Product: items/prices/outfits.
"return policy?" FAQ | "blue dress?" Product | "store hours?" FAQ | "sunny look?" Product
One word only: FAQ or Product. Query: {query}"""
```

### Creative/technical classifier: ~150 → ~60 tokens

```python
PROMPT_SIMPLIFIED = f"""Creative (outfit ideas/styling) or Technical (product info/availability)?
"outfit for wedding" creative | "blue jeans available?" technical
One word: creative or technical. Query: {query}"""
```

### FAQ lookup: ~1,200 → ~350 tokens

```python
# Full: inject all FAQ entries into the prompt
faq_layout = generate_faq_layout(faq)   # all 25 entries

# Simplified: semantic search → inject only top-5 relevant entries
results    = faq_collection.query.near_text(query, limit=5)
faq_5      = [x.properties for x in results.objects]
faq_5.reverse()                          # most relevant last (recency bias)
faq_layout = generate_faq_layout(faq_5)
```

### Metadata extraction: ~1,500 → 0 tokens

```python
# Full: call LLM to extract structured filters, then filtered semantic search
json_str, tokens = generate_metadata_from_query(query, values)
json_output      = parse_json_output(json_str)
filters          = get_filter_by_metadata(json_output)
results          = products_collection.query.near_text(query, filters=..., limit=20)

# Simplified: skip LLM entirely, direct semantic search
results = products_collection.query.near_text(query, limit=20)
tokens  = 0
```

### Token savings summary

| Step | Full | Simplified | Saving |
|------|------|------------|--------|
| Router | 200 | 80 | −60% |
| Creative/technical | 150 | 60 | −60% |
| FAQ lookup | 1,200 | 350 | −71% |
| Metadata extraction | 1,500 | 0 | −100% |
| Final answer | 800 | 800 | 0% |
| **Total (product query)** | **~2,650** | **~940** | **−65%** |

---

## 3. The `simplified` Flag Pattern

Add a boolean `simplified` parameter to every function in the pipeline. This allows A/B testing, gradual rollout, and easy rollback.

```python
def answer_query(
    query: str,
    model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    simplified: bool = False
) -> tuple[dict, int]:
    total_tokens = 0

    label, tokens = check_if_faq_or_product(query, simplified=simplified)
    total_tokens += tokens

    if label == "FAQ":
        kwargs = query_on_faq(query, simplified=simplified)
    elif label == "Product":
        kwargs, tokens = query_on_products(query, simplified=simplified)
        total_tokens += tokens
    else:
        kwargs = generate_params_dict(f"Answer from general knowledge: {query}", role="assistant")

    kwargs['model'] = model
    return kwargs, total_tokens

# Full pipeline
kwargs, prep = answer_query("Do you have waterproof jackets?", simplified=False)
result = generate_with_single_input(**kwargs)
print(f"Total tokens: {prep + result['usage']['total_tokens']}")  # ~2,650

# Simplified pipeline
kwargs, prep = answer_query("Do you have waterproof jackets?", simplified=True)
result = generate_with_single_input(**kwargs)
print(f"Total tokens: {prep + result['usage']['total_tokens']}")  # ~940
```

---

## 4. Observability with OpenTelemetry + Phoenix

### Core vocabulary

**Span** — a single, timed operation with a name, start/end time, and key-value attributes. The atomic unit of a trace.

**Trace** — a tree of spans representing one end-to-end request through the system. Spans nest to show parent-child relationships.

```
answer_query                        [chain]      450ms
  ├─ check_if_faq_or_product        [tool]        80ms
  │    └─ router_call               [llm]         75ms   80 tokens
  └─ query_on_products              [tool]       370ms
       ├─ decide_task_nature        [tool]        60ms
       │    └─ router_call          [llm]         55ms   60 tokens
       ├─ get_relevant_products     [retriever]   90ms
       └─ llm_call                  [llm]        220ms  800 tokens
```

### Setup

```python
import phoenix as px
from phoenix.otel import register
from opentelemetry.trace import Status, StatusCode

# 1. Launch Phoenix UI (available at http://localhost:6006)
session = px.launch_app()

# 2. Register tracer provider
tracer_provider = register(
    project_name="chatbot",
    endpoint="http://127.0.0.1:6006/v1/traces",
    auto_instrument=True    # auto-traces all OpenAI-compatible LLM calls
)

# 3. Get a tracer for manual spans
tracer = tracer_provider.get_tracer(__name__)
```

---

## 5. Span Kinds

| `openinference_span_kind` | Use for |
|---------------------------|---------|
| `"chain"` | Multi-step orchestration functions |
| `"tool"` | Helper functions (routing, formatting) |
| `"llm"` | LLM calls |
| `"retriever"` | Vector DB queries |

---

## 6. Instrumentation Patterns

### Decorator (simple)

```python
@tracer.chain
def format_context(results) -> str:
    return "".join(f"Q: {obj.properties['question']}\nA: {obj.properties['answer']}\n"
                   for obj in results.objects)

@tracer.tool
def generate_faq_layout(faq_list: list) -> str:
    return "".join(
        f"Question: {f['question']} Answer: {f['answer']}\n"
        for f in faq_list
    )
```

### Manual span (fine-grained control)

```python
def retrieve_faq(query: str, limit: int = 5):
    with tracer.start_as_current_span(
        "retrieve_faq_questions",
        openinference_span_kind="retriever"
    ) as span:
        span.set_input({"query": query, "limit": limit})
        try:
            results = faq_collection.query.near_text(query, limit=limit)

            # Log each retrieved document
            for i, doc in enumerate(results.objects):
                span.set_attribute(f"retrieval.documents.{i}.document.id",      str(doc.uuid))
                span.set_attribute(f"retrieval.documents.{i}.document.content",  str(doc.properties))
                span.set_attribute(f"retrieval.documents.{i}.document.metadata", str(doc.metadata))

            span.set_output({"count": len(results.objects)})
            span.set_status(Status(StatusCode.OK))
            return results
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise
```

### LLM call span with token tracking

```python
def call_llm_traced(kwargs: dict) -> dict:
    with tracer.start_as_current_span("llm_call", openinference_span_kind="llm") as span:
        span.set_input(kwargs)
        try:
            response = generate_with_single_input(**kwargs)
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
            raise
        else:
            # These attributes enable cost calculation in Phoenix
            span.set_attribute("llm.token_count.prompt",     response['usage']['prompt_tokens'])
            span.set_attribute("llm.token_count.completion", response['usage']['completion_tokens'])
            span.set_attribute("llm.token_count.total",      response['usage']['total_tokens'])
            span.set_attribute("llm.model_name",             response['model'])
            span.set_attribute("llm.provider",               "together.ai")
            span.set_output(response)
            span.set_status(Status(StatusCode.OK))
        return response
```

### Nested spans (full router example)

```python
def check_if_faq_or_product(query: str, simplified: bool = False):
    with tracer.start_as_current_span(
        "routing_faq_or_product", openinference_span_kind="tool"
    ) as outer_span:
        outer_span.set_input({"query": query, "simplified": simplified})

        PROMPT = build_router_prompt(query, simplified)
        kwargs = generate_params_dict(PROMPT, temperature=0, max_tokens=10)

        with tracer.start_as_current_span("router_llm_call", openinference_span_kind="llm") as llm_span:
            llm_span.set_input(kwargs)
            try:
                response = generate_with_single_input(**kwargs)
            except Exception as e:
                llm_span.record_exception(e)
                llm_span.set_status(Status(StatusCode.ERROR))
                raise
            else:
                llm_span.set_attribute("llm.token_count.prompt",     response['usage']['prompt_tokens'])
                llm_span.set_attribute("llm.token_count.completion", response['usage']['completion_tokens'])
                llm_span.set_attribute("llm.token_count.total",      response['usage']['total_tokens'])
                llm_span.set_attribute("llm.model_name",             response['model'])
                llm_span.set_output(response)
                llm_span.set_status(Status(StatusCode.OK))

        raw          = response['choices'][0]['message']['content']
        total_tokens = response['usage']['total_tokens']

        label = 'FAQ' if 'faq' in raw.lower() else 'Product' if 'product' in raw.lower() else 'undefined'
        outer_span.set_output({"label": label, "total_tokens": total_tokens})
        outer_span.set_status(Status(StatusCode.OK))
        return label, total_tokens
```

---

## 7. Auto-Instrumentation

With `auto_instrument=True`, all calls to `openai.chat.completions.create()` (and compatible wrappers like Together.ai) are traced automatically — no extra code needed.

```python
tracer_provider = register(
    project_name="my-rag-app",
    endpoint="http://127.0.0.1:6006/v1/traces",
    auto_instrument=True    # ← traces all OpenAI-compatible LLM calls
)
tracer = tracer_provider.get_tracer(__name__)

from openai import OpenAI
llm_client = OpenAI(api_key="...", base_url="https://api.together.xyz/v1")

# This call is automatically traced — no span code needed
response = llm_client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=[{"role": "user", "content": "What is RAG?"}]
)
```

---

## 8. Setting Up Cost Tracking in Phoenix

1. Navigate to `Settings → Models → Add Model`
2. Enter model name and pattern (e.g., `Qwen/Qwen3.5-9B`)
3. Set cost per million input tokens and output tokens
4. Phoenix calculates USD cost per trace automatically

After setup, every trace in the UI shows estimated cost alongside latency and token counts.

---

## 9. Minimal Traced RAG Pipeline

```python
@tracer.chain
def rag_pipeline(query: str) -> str:
    # Step 1: retrieve
    with tracer.start_as_current_span("retrieve", openinference_span_kind="retriever") as span:
        span.set_input(query)
        results = faq_collection.query.near_text(query, limit=5)
        for i, doc in enumerate(results.objects):
            span.set_attribute(f"retrieval.documents.{i}.document.content", str(doc.properties))
        span.set_status(Status(StatusCode.OK))

    # Step 2: format context
    context = "\n".join(
        f"Q: {obj.properties['question']}\nA: {obj.properties['answer']}"
        for obj in results.objects
    )

    # Step 3: generate (auto-traced if auto_instrument=True)
    prompt   = f"Answer using the context below.\nContext:\n{context}\nQuery: {query}"
    response = llm_client.chat.completions.create(
        model="Qwen/Qwen3.5-9B",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

answer = rag_pipeline("Can I return an item I bought by mistake?")
```

---

## 10. Production Design Principles

| Principle | How to apply |
|-----------|-------------|
| **Measure before cutting** | Trace all LLM calls; establish a token-per-query baseline before optimizing |
| **Cut the most expensive step first** | Metadata extraction (~1,500 tokens) is the highest-value target |
| **Preserve accuracy** | Test simplified prompts on a labeled query set before deploying |
| **Use cheaper models for routing** | Small models (Qwen/7B) for classification; large models (Llama/70B) for final answers |
| **Add observability from day one** | Tracing is cheap to add early; retroactively instrumenting a complex system is painful |
| **Progressive fallback on filters** | Never return empty results — relax filters iteratively until you have enough |
| **Most relevant context last** | LLMs exhibit recency bias; put the best retrieved document nearest the query |

---

## See Also

- [[RAG_Prompt_Engineering]] — the prompts that drive the pipeline optimized here
- [[RAG_Fundamentals]] — the full architecture this optimization applies to
