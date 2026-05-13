# LLM Parameters & Generation

## How Text Generation Works

An LLM processes the input sequence and outputs a **probability distribution** over its entire vocabulary — each token gets a score representing how likely it is to come next. Several parameters shape how the model samples from this distribution before selecting the next token.

Default (no parameters): **greedy decoding** — always pick the highest-probability token. Deterministic, but often repetitive and uncreative.

---

## Temperature

A scalar that reshapes the probability distribution before sampling.

$$\text{adjusted\_prob}(p_i) = \frac{\exp(\log(p_i) \,/\, T)}{\sum_j \exp(\log(p_j) \,/\, T)}$$

| Temperature | Effect on distribution | Effect on output |
|-------------|----------------------|-----------------|
| 0 | Collapses to argmax (deterministic) | Always the same answer |
| 0.3 | Sharpened — top tokens more dominant | Consistent, predictable |
| 0.7–1.0 | Unchanged (neutral) | Balanced creativity |
| 1.2–1.5 | Flattened — more uniform | More varied, occasional surprises |
| > 2.0 | Very flat — near-random | Often incoherent |

```python
query = "In one sentence, explain what RAG is."

# Deterministic — three calls return identical output
results = [generate_with_single_input(query, temperature=0) for _ in range(3)]

# Variable — three calls return different but coherent outputs
results = [generate_with_single_input(query, temperature=0.8) for _ in range(3)]
```

> **Important:** High temperature also reduces the probability of the stop token, so responses may run to `max_tokens` instead of stopping naturally. Always set a reasonable `max_tokens` when using high temperature.

---

## Top-p (Nucleus Sampling)

Restricts sampling to the **smallest set of tokens whose cumulative probability ≥ p**.

- `top_p = 0` → greedy (same as `temperature = 0`)
- `top_p = 0.5` → pick from the top tokens that together cover 50% probability mass
- `top_p = 1.0` → all tokens are candidates (probability distribution unchanged)

Unlike temperature, top_p doesn't reshape probabilities — it just **caps the pool** of eligible tokens.

```python
# Three calls → three different but coherent outputs
results = [generate_with_single_input(query, top_p=0.8) for _ in range(3)]
```

### Using Temperature + Top-p Together

These two parameters are complementary:
- **Temperature** reshapes the distribution (how flat or sharp)
- **Top-p** limits which tokens can be picked (how large the candidate pool)

```python
# High temperature makes distribution flat → could pick nonsense
# Low top_p caps the pool → prevents really bad tokens from being selected
generate_with_single_input(prompt, temperature=1.5, top_p=0.5)

# Result: varied output, but drawn only from plausible tokens
```

---

## Top-k

Restricts sampling to the top `k` most probable tokens, regardless of their cumulative probability.

- `top_k = 0` → no restriction (greedy by default)
- `top_k = 10` → pick randomly from the 10 most likely tokens at each step
- `top_k = 50` → wider candidate pool, more variety

```python
results = [generate_with_single_input(query, top_k=10) for _ in range(3)]
# Outputs will vary but stay among the most plausible completions
```

**Top-k vs top-p:** Top-p adapts to the distribution shape (picks fewer tokens when one is dominant, more when distribution is flat). Top-k is a fixed count regardless. Top-p is generally preferred.

---

## Repetition Penalty

Penalizes tokens that have already appeared in the generated text, discouraging repetitive phrasing.

- `1.0` → no penalty (default)
- `1.2` → mild; prevents redundant phrases
- `> 2.0` → strong; may distort text by penalizing common function words ("the", "and", "is")

```python
response = generate_with_single_input(
    "List healthy breakfast options.",
    repetition_penalty=1.2,
    max_tokens=300
)
```

---

## Parameter Recommendations by Task

| Task | Temperature | Top-p | Max Tokens | Notes |
|------|-------------|-------|------------|-------|
| Classification / routing | 0 | 0.1 | 1–10 | Must be deterministic |
| JSON / structured output | 0–0.3 | 0.7 | 500–1500 | Low randomness for valid JSON |
| Technical product queries | 0.3 | 0.7 | 500 | Precise, consistent |
| FAQ answers | 0.5 | 0.8 | 500 | Balanced |
| General conversation | 0.7 | 0.9 | 500 | Natural, engaging |
| Creative outfit suggestions | 1.0 | 0.9 | 500 | Variety and creativity |
| Creative writing / poems | 1.2 | 0.8 | 300 | Controlled creativity |

---

## The Parameter Dictionary Pattern

Build a `kwargs` dict to separate prompt construction from LLM execution. This enables routing: different query types get different parameter sets.

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

# Build → inspect → call
kwargs   = generate_params_dict("Suggest an outfit for a beach wedding.", temperature=1.0, top_p=0.9)
response = generate_with_single_input(**kwargs)
content  = response['content']
```

### Task-based parameter selection

```python
PARAMS_BY_TASK = {
    "creative":  {"temperature": 1.0, "top_p": 0.9},
    "technical": {"temperature": 0.3, "top_p": 0.7},
}

def get_params_for_task(task: str) -> dict:
    return PARAMS_BY_TASK.get(task, {"temperature": 0.5, "top_p": 0.8})  # fallback

# In the pipeline:
task   = decide_task_nature(query)          # → "creative" or "technical"
params = get_params_for_task(task)
kwargs = generate_params_dict(prompt, **params)
```

---

## Multi-turn Conversation (Chatbot)

Maintain conversation history by accumulating messages in a list and passing all of them on each call.

```python
def call_llm_with_context(
    prompt: str,
    context: list,
    role: str = "user",
    **kwargs
) -> dict:
    """Append user message, call LLM, append response, return response."""
    context.append({"role": role, "content": prompt})
    response = generate_with_multiple_input(context, **kwargs)
    context.append(response)   # {'role': 'assistant', 'content': '...'}
    return response

# Initialize conversation
context = [
    {"role": "system",    "content": "You are a helpful fashion assistant."},
    {"role": "assistant", "content": "How can I help you today?"}
]

# Turn 1
r = call_llm_with_context("Suggest a summer outfit.", context=context)
print(r['content'])

# Turn 2 — the model remembers the previous exchange
r = call_llm_with_context("Can you make it more formal?", context=context)
print(r['content'])
```

After two turns, `context` looks like:
```python
[
    {"role": "system",    "content": "You are a helpful fashion assistant."},
    {"role": "assistant", "content": "How can I help you today?"},
    {"role": "user",      "content": "Suggest a summer outfit."},
    {"role": "assistant", "content": "I recommend a linen shirt..."},
    {"role": "user",      "content": "Can you make it more formal?"},
    {"role": "assistant", "content": "In that case, a light blazer..."},
]
```

---

## Reading Token Usage (Module 5 Format)

In production RAG (Module 5), the helper returns the full OpenAI-format response for cost tracking:

```python
result = generate_with_single_input("What is RAG?")

content           = result['choices'][0]['message']['content']
prompt_tokens     = result['usage']['prompt_tokens']
completion_tokens = result['usage']['completion_tokens']
total_tokens      = result['usage']['total_tokens']
model_name        = result['model']
```

---

## Effect Summary

```
Greedy decoding (temp=0, top_p=0):
  "The quick brown fox" → "jumps" (always, deterministic)

top_p=0.8, temp=1.0 (nucleus sampling):
  "The quick brown fox" → "jumps" | "leaps" | "runs" (varies each call)

top_p=0.8, temp=1.5 (high temperature):
  "The quick brown fox" → "jumps" | "dashes" | "sprints" | rare surprises

top_p=0.1, temp=1.5 (constrained nucleus):
  "The quick brown fox" → "jumps" (nucleus is tiny, even with high temp)

repetition_penalty=1.3:
  Forces variety in word choice; prevents "The cat. The cat. The cat."
```

---

## See Also

- [[RAG_Prompt_Engineering]] — using these parameters in a routing architecture
- [[RAG_Production]] — tracking token costs across all LLM calls
