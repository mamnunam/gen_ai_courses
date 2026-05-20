---
tags: [pattern, agentic-ai, reflection]
aliases: [reflection, self-critique, draft-revise]
---

# Reflection Pattern

The simplest agentic loop: **generate → critique → revise**. An LLM produces a first draft (V1), a second LLM call critiques it, and a third call rewrites V1 based on the critique to produce V2.

Reflection turns a stochastic one-shot generation into a small workflow where the model effectively "checks its own work." It's the cheapest agentic improvement you can add to a single LLM call.

> [!tip]+ When to use
> - Creative or open-ended tasks where the first draft is usually serviceable but rarely great (essays, marketing copy, design choices).
> - Code or query generation where syntactic correctness ≠ semantic correctness — but only the *text-level* variant; for semantic correctness you need [[External Feedback Reflection]].
> - Tasks where you can afford 2-3× the latency and tokens for noticeably better output.

## Three-step shape

```python
def generate_draft(topic, model="openai:gpt-4o"):
    prompt = f"Write a complete draft essay on: {topic}"
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    ).choices[0].message.content

def reflect_on_draft(draft, model="openai:o4-mini"):
    prompt = f"""Provide constructive feedback on this essay.
    Address structure, clarity, argument strength, and writing style.
    Draft:
    {draft}"""
    return client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=1.0
    ).choices[0].message.content

def revise_draft(original_draft, reflection, model="openai:gpt-4o"):
    prompt = f"""Improve this essay based on the feedback.
    Address all issues, improve clarity and flow, return only the final essay.
    Original: {original_draft}
    Feedback: {reflection}"""
    return client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=1.0
    ).choices[0].message.content
```

The whole pipeline is just three function calls:

```python
draft = generate_draft(essay_prompt)
feedback = reflect_on_draft(draft)
revised = revise_draft(draft, feedback)
```

## Variants

The pattern has several flavors that warrant their own atomic notes:

- **Text-only reflection** — model critiques the V1 text directly (this note). Works for stylistic/structural issues but misses semantic errors invisible from the text alone.
- [[External Feedback Reflection]] — feed the *execution result* (DataFrame, image, runtime error) into the critic. Required when the bug is only visible after running the code.
- **Image-grounded reflection** — pass the rendered output (chart, generated image) to a multimodal critic via [[Multimodal Image Input]].

> [!tip]+ Tips that recur in the course
> - Use [[Mixed-Model Strategy]]: cheap model for generation, stronger reasoning model (o4-mini, gpt-4.1) for critique.
> - Force a strict output contract for the critique step — see [[Structured JSON Outputs]] — so the revise step has clean inputs.
> - `temperature=1.0` for generation/revision (creative); the critic can also use 1.0 since you want diverse critiques.

> [!failure]+ Failure mode to remember
> Pure text-only reflection cannot catch errors where the SQL/code is *syntactically valid but semantically wrong* — for example summing a signed `qty_delta` column without `ABS()`. The reviewer reads the query, says "looks fine," and confirms a wrong answer. This is exactly the case for [[External Feedback Reflection]].

## Seen in

- [[C1M2 Assignment]] — pure text-only reflection on essays (the canonical three-step pattern).
- [[M2 UGL 1]] — reflection on a chart image (image-grounded).
- [[M2 UGL 2]] — reflection on SQL, both text-only (fails) and execution-grounded (succeeds).
- [[C1M3 Assignment]] — reflection on a research report, with JSON-structured critique output.
