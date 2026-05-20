---
tags: [pattern, agentic-ai, reflection, grounding]
aliases: [execution-grounded reflection, reflection with feedback]
---

# External Feedback Reflection

A variant of the [[Reflection Pattern]] where the critic doesn't just see the V1 **artifact** (text, code, SQL) — it also sees the **result of running it**. The execution output becomes external feedback that grounds the critique in reality instead of the model's belief about reality.

This is the single most important refinement of basic reflection. It catches errors that are invisible from the artifact alone.

## The motivating failure

In `M2_UGL_2`, the user asks "which color of product has the highest total sales?" The model generates SQL that correctly uses `SUM(qty_delta * unit_price)` grouped by color. The critic reads the SQL and says "looks correct." But when you actually run it, every total is **negative** — because in this schema, sales events store `qty_delta < 0` (inventory leaving). The sign bug is invisible from the SQL text. It's only visible from the DataFrame.

> The SQL text alone looked fine. The actual execution output revealed a sign error.

## Implementation

The difference from plain reflection is one extra input: the result of executing V1.

```python
def refine_sql_external_feedback(question, sql_query, df_feedback, schema, model):
    prompt = f"""
    You are a SQL reviewer and refiner.

    User asked: {question}
    Original SQL: {sql_query}
    SQL Output:
    {df_feedback.to_markdown(index=False)}   # ← the critical addition
    Table Schema: {schema}

    Step 1: Briefly evaluate if the SQL output answers the user's question.
    Step 2: If improvement is needed, provide a refined SQL query.

    Return strict JSON: {{"feedback": "...", "refined_sql": "..."}}
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
    )
    # parse JSON, return (feedback, refined_sql)
```

The full workflow:

```python
def run_sql_workflow(db_path, question, model_gen, model_eval):
    schema = utils.get_schema(db_path)
    sql_v1 = generate_sql(question, schema, model_gen)
    df_v1  = utils.execute_sql(sql_v1, db_path)          # ← run V1
    feedback, sql_v2 = refine_sql_external_feedback(
        question=question, sql_query=sql_v1,
        df_feedback=df_v1,                                # ← feed result back
        schema=schema, model=model_eval,
    )
    df_v2 = utils.execute_sql(sql_v2, db_path)
    return df_v2
```

## Forms of external feedback

The "feedback" can be anything observable that grounds the critic:

| Artifact type     | Feedback form                                       | Example                  |
| ----------------- | --------------------------------------------------- | ------------------------ |
| SQL               | Result DataFrame (as markdown table)                | `M2_UGL_2`               |
| Python plot code  | Rendered image (base64) via [[Multimodal Image Input]] | `M2_UGL_1`               |
| Tool-calling code | stdout, exceptions, before/after table snapshots    | `M5_UGL_1_R`             |
| Web search        | List of returned URLs vs. preferred-domain rubric   | `M4_UGL_1`               |

> [!info]+ Why it matters
> Reflection without grounding is just the model talking to itself with the same priors. Reflection with execution output forces the model to **reconcile its belief with observed reality**. This is the same idea that distinguishes ReAct from pure chain-of-thought.

> [!warning]+ Trade-offs
> - You must be able to safely execute V1 — see [[Sandboxed Execution]] for the standard mitigation.
> - Adds latency: now you have generate → execute → reflect → execute V2.
> - For destructive operations (mutations, API writes), executing V1 may have side effects. Use dry-run or transactional rollback.

## Seen in

- [[M2 UGL 2]] — SQL refinement with DataFrame feedback (the canonical example).
- [[M2 UGL 1]] — chart code refinement with rendered-image feedback.
- [[M5 UGL 1 R]] — before/after table snapshots as feedback after [[Code as Plan]] execution.
- [[M4 UGL 1]] — preferred-domain ratio as machine-readable feedback for [[Component-Level Evaluation]].
