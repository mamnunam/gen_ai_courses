# Prompt Engineering

## Core Patterns

| Pattern | When to use |
|---------|-------------|
| **Few-shot classification** | Route queries to the right handler; classify intent |
| **Structured JSON output** | Extract metadata, structured data, schemas |
| **Context injection** | Standard RAG prompt with retrieved documents |
| **Progressive filter relaxation** | Handle over-filtered database queries |
| **Simplified prompts** | Reduce token cost while preserving accuracy |

---

## 1. Few-Shot Classification

Instruct the LLM to output a fixed label by:
1. Explaining the two (or more) categories clearly
2. Providing labeled examples (few-shot) covering edge cases
3. Restricting max tokens (`max_tokens=1` or `10`)
4. Using `temperature=0` for determinism

### Pattern

```python
def classify(query: str, prompt: str, valid_labels: tuple) -> str:
    kwargs   = generate_params_dict(prompt, temperature=0, max_tokens=10)
    response = generate_with_single_input(**kwargs)
    label    = response['content'].lower()

    for valid in valid_labels:
        if valid.lower() in label:
            return valid
    return "undefined"
```

---

## 2. Query Router: FAQ vs. Product

Classifies every incoming query before deciding which RAG path to use.

```python
def check_if_faq_or_product(query: str, simplified: bool = False) -> tuple[str, int]:
    """
    Returns (label, total_tokens).
    label is 'FAQ', 'Product', or 'undefined'.
    """
    if not simplified:
        # Full prompt — higher accuracy, ~200 tokens
        PROMPT = f"""Label the following instruction for a clothing store chatbot.
FAQ: questions about policies, returns, shipping, contact, store hours.
Product: questions about specific items, prices, colors, outfit creation.

Examples:
Is there a refund for incorrectly bought clothes? Label: FAQ
Where are your stores located? Label: FAQ
Tell me about the cheapest T-shirts you have. Label: Product
Do you have blue T-shirts under 100 dollars? Label: Product
What are the available sizes for t-shirts? Label: FAQ
Give me ideas for a sunny look. Label: Product
How can I find promotions? Label: FAQ
Create a look for a wedding party. Label: Product

Return only one word: FAQ or Product.
Query: {query}"""

    else:
        # Simplified prompt — ~80 tokens, same accuracy
        PROMPT = f"""Classify for a clothing store.
FAQ: policies/shipping/contact/returns. Product: items/prices/outfits/styles.
Examples: "return policy?" FAQ | "blue dress?" Product | "store hours?" FAQ | "sunny look?" Product
One word only: FAQ or Product.
Query: {query}"""

    kwargs   = generate_params_dict(PROMPT, temperature=0, max_tokens=10)
    response = generate_with_single_input(**kwargs)

    raw          = response['choices'][0]['message']['content']
    total_tokens = response['usage']['total_tokens']

    if 'faq'     in raw.lower(): return 'FAQ',       total_tokens
    if 'product' in raw.lower(): return 'Product',   total_tokens
    return 'undefined', total_tokens
```

---

## 3. Sub-Router: Creative vs. Technical

Once a query is classified as Product, determine whether it needs a creative response (outfit ideas) or a technical one (exact product lookup). This controls the LLM parameters used downstream.

```python
def decide_task_nature(query: str, simplified: bool = True) -> tuple[str, int]:
    """
    Returns ('creative' | 'technical', total_tokens).
    """
    if not simplified:
        PROMPT = f"""Decide if this query requires creativity or technical information.
Creative: creating outfits, making style suggestions, composing looks.
Technical: listing products, checking prices, finding available items.

Examples:
Suggest a look for a nightclub. Label: creative
What blue dresses do you have? Label: technical
Give me three summer T-shirts. Label: technical
Create a look for a beach wedding. Label: creative
What matches a green T-shirt? Label: creative

Query: {query}. Output one word: creative or technical."""
    else:
        PROMPT = f"""Creative (outfit ideas/styling) or Technical (product info/availability)?
"outfit for wedding" creative | "blue jeans available?" technical
One word: creative or technical.
Query: {query}"""

    kwargs   = generate_params_dict(PROMPT, temperature=0, max_tokens=1)
    response = generate_with_single_input(**kwargs)
    label    = response['choices'][0]['message']['content']
    return label, response['usage']['total_tokens']
```

---

## 4. FAQ Prompt

```python
def generate_faq_layout(faq_list: list) -> str:
    """Format a list of FAQ dicts into a readable string for the prompt."""
    return "".join(
        f"Question: {f['question']} Answer: {f['answer']} Type: {f['type']}\n"
        for f in faq_list
    )

def query_on_faq(query: str, simplified: bool = False, **kwargs) -> dict:
    """
    Returns a generate_params_dict ready to be passed to generate_with_single_input.
    simplified=True uses Weaviate semantic search to include only the top-5 FAQ entries.
    """
    if not simplified:
        # Inject full FAQ — ~1,200 tokens
        faq_layout = generate_faq_layout(faq)

    else:
        # Retrieve top-5 relevant FAQ entries — ~350 tokens
        results = faq_collection.query.near_text(query, limit=5)
        relevant = [x.properties for x in results.objects]
        relevant.reverse()   # least relevant first → most relevant near end of prompt (recency bias)
        faq_layout = generate_faq_layout(relevant)

    PROMPT = (
        f"Answer the query using the FAQ below for a clothing store. "
        f"Don't mention you have access to an FAQ.\n"
        f"<FAQ>\n{faq_layout}\n</FAQ>\n"
        f"Query: {query}"
    )
    return generate_params_dict(PROMPT, **kwargs)
```

---

## 5. Structured JSON Output

### Prompt-based JSON generation

Force JSON output by:
- Describing the exact schema in the prompt
- Providing a concrete example using escaped braces in f-strings (`{{` → `{`)
- Setting `temperature=0` and `max_tokens` high enough for the JSON

```python
def generate_metadata_from_query(query: str, values: dict) -> str:
    """
    Ask the LLM to extract structured metadata from a natural-language query.
    Returns a JSON string with filter values for the product database.
    """
    PROMPT = f"""Given this clothing store query, generate a JSON for database filtering.
Possible values per feature: {values}

Required keys: gender, masterCategory, articleType, baseColour, price, usage, season.
Rules:
- price must be {{"min": number, "max": number}} (use "inf" if no upper bound, 0 if no lower)
- All other values must be lists (e.g. ["Men"])
- Only include values that exist in the possible values above
- If a feature is unrestricted, use ["Any"]

Example output:
{{
  "gender":         ["Women"],
  "masterCategory": ["Apparel"],
  "articleType":    ["Dresses"],
  "baseColour":     ["Blue"],
  "price":          {{"min": 0, "max": "inf"}},
  "usage":          ["Formal"],
  "season":         ["All seasons"]
}}

Query: {query}
Return only the JSON, nothing else."""

    response = generate_with_single_input(PROMPT, temperature=0, max_tokens=1500)
    return response['choices'][0]['message']['content']
```

### Parsing JSON output

```python
import json

def parse_json_output(llm_output: str) -> dict | None:
    try:
        cleaned = (llm_output
                   .replace("\n", "")
                   .replace("'", "")
                   .replace("}}", "}")
                   .replace("{{", "{"))
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON parse failed: {e}")
        return None
```

### Pydantic schema (OpenAI-compatible structured output)

```python
from pydantic import BaseModel, Field

class ProductFilter(BaseModel):
    gender:         list[str] = Field(description="Target gender(s)")
    articleType:    list[str] = Field(description="Types of garment")
    baseColour:     list[str] = Field(description="Colors")
    season:         list[str] = Field(description="Seasons")

response_format = {
    "type":   "json_schema",
    "schema": ProductFilter.model_json_schema()
}

result = generate_with_multiple_input(messages, response_format=response_format)
parsed = ProductFilter.model_validate_json(result['content'])
```

---

## 6. Metadata-Filtered Product Search

Convert the LLM-generated JSON into Weaviate filters, with **progressive relaxation** when filters are too strict.

```python
from weaviate.classes.query import Filter

def get_filter_by_metadata(json_output: dict | None) -> list | None:
    if not json_output:
        return None

    valid_keys = ('gender', 'masterCategory', 'articleType', 'baseColour', 'price', 'usage', 'season')
    filters = []

    for key, value in json_output.items():
        if key not in valid_keys:
            continue
        if key == 'price':
            min_p = value.get('min', 0)
            max_p = value.get('max', 'inf')
            if min_p and min_p > 0:
                filters.append(Filter.by_property(key).greater_than(min_p))
            if max_p and max_p != 'inf':
                filters.append(Filter.by_property(key).less_than(max_p))
        else:
            if value and value != ['Any']:
                filters.append(Filter.by_property(key).contains_any(value))

    return filters

def get_relevant_products(query: str, products_collection, simplified: bool = False):
    if simplified:
        # Skip LLM call entirely — direct semantic search
        results = products_collection.query.near_text(query, limit=20)
        return results.objects, 0   # (results, tokens_used)

    # Full path: LLM extracts metadata → filter → semantic search
    json_str, tokens = generate_metadata_from_query(query, values), ...
    json_output      = parse_json_output(json_str)
    filters          = get_filter_by_metadata(json_output)

    if not filters:
        return products_collection.query.near_text(query, limit=20).objects, tokens

    results = products_collection.query.near_text(
        query, filters=Filter.all_of(filters), limit=20
    )

    # Progressive relaxation if too few results
    importance_order = ['baseColour', 'masterCategory', 'usage', 'season', 'articleType', 'gender']
    if len(results.objects) < 10:
        for i in range(len(importance_order)):
            relaxed = [f for f in filters if f.target not in importance_order[:i+1]]
            results = products_collection.query.near_text(
                query, filters=Filter.all_of(relaxed), limit=20
            )
            if len(results.objects) >= 5:
                break

    return results.objects, tokens
```

---

## 7. Product Context Formatting

```python
def generate_items_context(results: list) -> str:
    return "".join(
        f"Product ID: {p['product_id']}. "
        f"Name: {p['productDisplayName']}. "
        f"Category: {p['masterCategory']}. "
        f"Usage: {p['usage']}. Gender: {p['gender']}. "
        f"Type: {p['articleType']}. Color: {p['baseColour']}. "
        f"Season: {p['season']}.\n"
        for item in results
        for p in [item.properties]
    )
```

---

## 8. Full Chatbot Architecture

```
User Query
    │
    ▼
check_if_faq_or_product(query)         ← temp=0, max_tokens=10
    │
    ├── "FAQ" ──────────────────────────────────────────────────────────────┐
    │                                                                        │
    │   query_on_faq(query, simplified=True)                                │
    │     └─ faq_collection.query.near_text(query, limit=5)                 │
    │     └─ generate_faq_layout(top_5_results)                             │
    │     └─ generate_params_dict(FAQ_PROMPT)           ─────────────────── │
    │                                                                        ▼
    └── "Product" ─────────────────────────────────────────────────────────┐ │
                                                                           │ │
        decide_task_nature(query)   ← temp=0, max_tokens=1                │ │
            │                                                              │ │
            ├── "creative"  → params = {temp:1.0, top_p:0.9}             │ │
            └── "technical" → params = {temp:0.3, top_p:0.7}             │ │
                                                                           │ │
        get_relevant_products(query, simplified=True)                      │ │
            └─ products_collection.query.near_text(query, limit=20)        │ │
                                                                           │ │
        generate_items_context(products)                                   │ │
            └─ format as "Product ID: X. Name: Y. ..."                    │ │
                                                                           │ │
        generate_params_dict(PRODUCT_PROMPT, **params)  ───────────────── │ │
                                                                           │ │
    ┌─────────────────────────────────────────────────────────────────────┘ │
    │                                                                        │
    ▼                                                                        │
kwargs['model'] = "meta-llama/Llama-3.3-70B-Instruct-Turbo"   ◄────────────┘
generate_with_single_input(**kwargs)
    └─ Final answer delivered to user
```

### Code

```python
def query_on_products(query: str, simplified: bool = False) -> tuple[dict, int]:
    total_tokens = 0

    task_nature, tokens = decide_task_nature(query, simplified=simplified)
    total_tokens += tokens

    params   = get_params_for_task(task_nature)
    products, tokens = get_relevant_products(query, products_collection, simplified)
    total_tokens += tokens

    context = generate_items_context(products)
    PROMPT  = (
        f"Answer the clothing store query using the available products below.\n"
        f"Include the Product ID in your answer. Use at most 5 products.\n"
        f"PRODUCTS:\n{context}\n"
        f"QUERY: {query}"
    )
    return generate_params_dict(PROMPT, role="assistant", **params), total_tokens

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
        # Fallback: let the LLM answer without RAG context
        kwargs = generate_params_dict(
            f"Answer this question from your general knowledge: {query}",
            role="assistant"
        )

    kwargs['model'] = model
    return kwargs, total_tokens

# Usage
kwargs, prep_tokens = answer_query("Do you have waterproof jackets?", simplified=True)
result      = generate_with_single_input(**kwargs)
answer      = result['choices'][0]['message']['content']
total_cost  = prep_tokens + result['usage']['total_tokens']
```

---

## 9. Context Injection Template

Standard RAG prompt structure:

```python
PROMPT_TEMPLATE = """Answer the user query below. Additional information is provided.
The context is from 2024 — use it alongside your general knowledge. Don't rely on it exclusively.
Context is ordered by relevance (most relevant last).

Query: {query}
Context:
{context}"""
```

Key design choices:
- **Most relevant context at the end** — LLMs exhibit recency bias; the last thing in the context has more influence.
- **Don't say "only use the context"** — this over-constrains the model when context is incomplete.
- **Don't say "you have access to a database"** — this breaks the illusion and can confuse users.

---

## See Also

- [[RAG_LLM_Parameters]] — temperature, top_p, and multi-turn conversation
- [[RAG_Weaviate]] — Weaviate query API used inside these functions
- [[RAG_Production]] — measuring and reducing the token cost of these prompts
