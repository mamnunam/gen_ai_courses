---
tags: [pattern, agentic-ai, planning, code-execution]
aliases: [code-as-action, planning with code execution, plan-as-code]
---

# Code as Plan

A planning pattern where the LLM writes **Python code that is itself the plan** — comments describe the steps, the code carries them out. Instead of producing a JSON list of step descriptions that you then execute with bespoke tools, you let Python (plus a focused toolbox like TinyDB) be the substrate for multi-step logic.

> *Andrew Ng (Module 5): "Let the model write code that becomes the plan itself."*

## The intuition

Tool-based planning produces brittle pipelines: every new operation needs a new tool, and the LLM has to compose them via the wire protocol. Code-as-plan inverts this — the LLM already knows Python, so let it filter, compute, branch, and mutate state in code that you then execute in a controlled namespace.

This trades a growing tool surface for a single, expressive substrate. It's appropriate when the task involves:

- Data operations the model already knows how to express (filter, group, aggregate, mutate).
- Multiple conditionally-related steps where a tool chain would be awkward.
- Logic that benefits from local variables, control flow, and intermediate computations.

## The contract

The pattern requires three pieces in concert:

1. **A prompt that constrains the code** — schema, available variables, safety rules, output convention. See [[Schema Injection Prompting]].
2. **A tag-based extraction** — the model wraps its code in `<execute_python>...</execute_python>`. See [[Tag-Based Code Extraction]].
3. **An executor** — runs the code in a controlled namespace with pre-bound table/helper objects. See [[Sandboxed Execution]].

## The prompt skeleton (M5_UGL_1_R)

The prompt is the heavy lifter. It does five things:

```
You are a senior data assistant. PLAN BY WRITING PYTHON CODE USING TINYDB.

Database Schema & Samples (read-only):
{schema_block}                                      # ← injected schema

Execution Environment (already imported/provided):
- Variables: db, inventory_tbl, transactions_tbl    # ← pre-bound objects
- Helpers: get_current_balance(tbl), next_transaction_id(tbl, prefix="TXN")
- Natural language: user_request: str

PLANNING RULES (critical):
- Derive ALL filters/parameters from user_request. Do NOT hard-code values.
- Build TinyDB queries dynamically. If a constraint isn't in user_request, don't apply it.
- Be conservative: if intent is ambiguous, do read-only (DRY RUN).

TRANSACTION POLICY (hard):
- One transaction row PER ITEM (no aggregated multi-item rows).
- For each item: compute line total, insert ONE transaction, update balance, update stock.

HUMAN RESPONSE REQUIREMENT (hard):
- Set `answer_text` (str) — short, customer-friendly. This is the only user-facing message.

OUTPUT CONTRACT:
- Return ONLY executable Python between:
  <execute_python>
  # your python
  </execute_python>
```

The convention variables (`answer_text`, `answer_rows`, `answer_json`) are the **plan→executor handoff**: the executor reads them back from the namespace after running the code.

## Generate → execute

```python
def generate_llm_code(prompt, *, inventory_tbl, transactions_tbl,
                     model="gpt-4.1-mini", temperature=0.2):
    schema_block = inv_utils.build_schema_block(inventory_tbl, transactions_tbl)
    final_prompt = PROMPT.format(schema_block=schema_block, question=prompt)
    resp = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[
            {"role": "system", "content": "You write safe, well-commented TinyDB code."},
            {"role": "user", "content": final_prompt},
        ],
    )
    return resp.choices[0].message.content   # full content, tags included
```

Then the executor (see [[Sandboxed Execution]]) extracts and runs the code with TinyDB tables in scope.

## End-to-end customer-service agent

```python
def customer_service_agent(question, *, db, inventory_tbl, transactions_tbl,
                           model="o4-mini", temperature=1.0, reseed=False):
    if reseed:
        inv_utils.create_inventory(); inv_utils.create_transactions()

    full_content = generate_llm_code(question,
        inventory_tbl=inventory_tbl, transactions_tbl=transactions_tbl,
        model=model, temperature=temperature)

    exec_res = execute_generated_code(full_content,
        db=db, inventory_tbl=inventory_tbl, transactions_tbl=transactions_tbl,
        user_request=question)

    return {"full_content": full_content, "exec": exec_res}
```

Real example prompts that work end-to-end:

- "Do you have any round sunglasses in stock that are under $100?" → read-only query.
- "Return 2 Aviator sunglasses I bought last week." → mutation (stock + transaction row).
- "I want to buy 3 pairs of classic sunglasses and 1 pair of aviator." → multi-item, one transaction per item.

> [!info]+ Why this beats tool-soup
> The lab calls out three reasons explicitly:
>
> - **No bespoke-tool tax.** New query shapes don't require new tools.
> - **The model already knows Python.** Less prompt engineering for orchestration; more for safety constraints.
> - **Transparent execution.** You see exactly what code ran, plus before/after snapshots of mutated state.

> [!warning]+ Trade-offs and risks
> - **Safety.** Arbitrary code execution is dangerous; the pattern is only as safe as the [[Sandboxed Execution]] enforcing the namespace and import policy.
> - **Determinism.** Code that the model writes is not deterministic across runs. You need before/after snapshots to audit.
> - **Complex semantics belong in the prompt.** Things like "one transaction per item" or "validate stock before mutating" must be in the prompt — the LLM won't infer business rules.

## Seen in

- [[M5 UGL 1 R]] — the canonical customer-service-agent implementation over TinyDB.

## Related

- [[Schema Injection Prompting]] — how the model learns the data shape.
- [[Tag-Based Code Extraction]] — pulling the code back out.
- [[Sandboxed Execution]] — running it without blowing up the world.
- [[Planner-Executor Architecture]] — the alternative: JSON plan + agent registry.
