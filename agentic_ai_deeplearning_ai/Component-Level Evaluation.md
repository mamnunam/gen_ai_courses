---
tags: [concept, agentic-ai, evaluation]
aliases: [component eval, per-step evaluation, sub-pipeline eval]
---

# Component-Level Evaluation

Evaluating **one component** of an agentic pipeline against ground truth, in isolation, instead of evaluating the whole pipeline end-to-end. Faster feedback, cheaper iteration, less noise — and you can pinpoint which component is the bottleneck.

> *Andrew Ng, M4:* "If the problem lies in web search (usually the first step), rerunning the entire pipeline every time can be expensive and noisy. Small improvements may be hidden by randomness from later components. Component-level evals give you a clearer signal."

## The motivating problem

The full research-agent pipeline is **search → draft → reflect → format**. If you want to know whether *search* is improving when you tweak its prompt, running the full pipeline:

- Costs 4× the LLM calls per iteration.
- Lets downstream randomness (drafting, reflection) drown out the signal from search.
- Doesn't tell you cleanly *where* the quality is changing.

Evaluating just `find_references` against a per-example ground truth (preferred-domain list) fixes all three.

## Where in the eval taxonomy

Andrew's two-axis taxonomy: **objective vs. subjective** × **per-example vs. corpus-level**.

Component-level + preferred-domain is in the **upper-left quadrant**: objective + per-example. The ground truth is a deterministic, code-checkable rubric — for each query, the set of preferred domains is fixed.

## The canonical example (M4_UGL_1)

The component is `find_references` — a search-only agent that uses arxiv/tavily/wikipedia tools and returns text containing URLs:

```python
def find_references(task, model="openai:gpt-4o", return_messages=False):
    """Perform a research task using external tools."""
    prompt = f"""
    You are a research function with access to:
    - arxiv_tool: academic papers
    - tavily_tool: general web search
    - wikipedia_tool: encyclopedic summaries

    Task: {task}
    Today is {datetime.now().strftime('%Y-%m-%d')}.
    """.strip()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        tools=[research_tools.arxiv_search_tool,
               research_tools.tavily_search_tool,
               research_tools.wikipedia_search_tool],
        tool_choice="auto",
        max_turns=5,
    )
    return response.choices[0].message.content
```

## The eval function

A pure Python rubric — no LLM-as-judge, no human review:

```python
TOP_DOMAINS = {
    "wikipedia.org", "nature.com", "science.org", "arxiv.org",
    "nasa.gov", "mit.edu", "stanford.edu", "harvard.edu",
    "ieee.org", "acm.org", "neurips.cc", "icml.cc", "openreview.net",
    # ...
}

def evaluate_tavily_results(TOP_DOMAINS, raw: str, min_ratio=0.4):
    """Returns (pass/fail, markdown_report)."""
    url_pattern = re.compile(r'https?://[^\s\]\)>\}]+', flags=re.IGNORECASE)
    urls = url_pattern.findall(raw)

    if not urls:
        return False, "No URLs detected."

    total = len(urls)
    preferred = sum(
        1 for url in urls
        if any(td in url.split("/")[2] for td in TOP_DOMAINS)
    )
    ratio = preferred / total
    flag = ratio >= min_ratio

    report = f"""
### Evaluation — Tavily Preferred Domains
- Total results: {total}
- Preferred: {preferred}
- Ratio: {ratio:.2%}
- Threshold: {min_ratio:.0%}
- Status: {"✅ PASS" if flag else "❌ FAIL"}
"""
    return flag, report
```

## What makes this a good eval

- **Objective** — same input always produces the same flag.
- **Cheap** — no LLM call to evaluate; just regex + set lookup.
- **Per-example ground truth** — each query has its own preferred-domain set (you can vary it per topic).
- **Actionable** — a FAIL tells you the search step is the problem, not the drafting step.

## Designing your own component eval

For any pipeline component, ask:

1. What is the output of this component? (URLs, JSON, SQL, code, text)
2. What does "correct enough" look like? (a list, a rubric, a regex, a structural check)
3. Can I assert it without an LLM? (preferred — cheap and reproducible)
4. If not, can I use a stronger LLM as judge? (LLM-as-judge — subjective but reproducible)

The lab calls out that you'd typically build a small eval set of ~10 prompts with their own preferred-domain lists, then run the component over them and average pass rates.

> [!warning]+ Trade-offs
> - **Misses interaction bugs.** A component can pass in isolation but break in context (e.g., search returns good URLs but in a format the drafting step can't parse). You still need occasional end-to-end runs.
> - **Designed for non-stochastic checks.** If the component is highly stochastic, you may need many samples per prompt to get a stable metric.
> - **Specific to objective rubrics.** For "is this report well-written?" you'd need [[Mixed-Model Strategy]] + LLM-as-judge.

## Seen in

- [[M4 UGL 1]] — the canonical example: preferred-domain check for `find_references`.

## Related

- [[Reflection Pattern]] — also a form of self-evaluation, but in-line not as a separate eval.
- [[External Feedback Reflection]] — execution output as a feedback signal (related to component eval).
- [[Research Tools Module]] — what `find_references` wraps.
