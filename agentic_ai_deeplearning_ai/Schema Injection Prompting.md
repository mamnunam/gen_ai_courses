---
tags: [technique, agentic-ai, prompt-engineering, data]
aliases: [schema in prompt, dataset context prompt]
---

# Schema Injection Prompting

Telling the LLM exactly what your data looks like — column names, types, value ranges, samples — by **embedding the schema directly in the prompt** before asking it to operate on the data. Without this, the model hallucinates field names; with it, the model's queries/code work on the first try far more often.

Comes up everywhere the model is generating SQL, pandas code, or TinyDB queries against your data.

> [!info]+ Why it matters
> The model can't see your database or DataFrame. If you ask "find all sales in Q1 2024" and the model guesses your column is named `date_of_sale` when it's actually `ts`, every query is broken. Schema injection eliminates the guess.
>
> It also tells the model:
>
> - **Types** — `date` is `datetime64`, not a string; don't concatenate it.
> - **Already-computed columns** — `quarter` exists as int, use it directly instead of computing `df['date'].dt.quarter`.
> - **Sample values** — `action` is one of `insert / restock / sale / price_update`.

## Three forms

### 1. Inline schema string (M2_UGL_2 — SQL)

```python
schema = """
Table name: transactions
id (INTEGER)
product_id (INTEGER)
product_name (TEXT)
brand (TEXT)
category (TEXT)
color (TEXT)
action (TEXT)
qty_delta (INTEGER)
unit_price (REAL)
notes (TEXT)
ts (DATETIME)
"""

prompt = f"""
You are a SQL assistant. Given the schema and the user's question,
write a SQL query for SQLite.

Schema:
{schema}

User question:
{question}

Respond with the SQL only.
"""
```

### 2. Column list with type notes and rules (M2_UGL_1 — pandas)

```text
The code should create a visualization from a DataFrame 'df' with these columns:
- date   (datetime64 — already parsed; use df['date'].dt.year, etc.)
- time   (string, HH:MM — do NOT concatenate or combine with the date column)
- cash_type (string: 'card' or 'cash')
- price (number)
- coffee_name (string)
- quarter (int, 1–4 — already computed, use directly)
- month  (int, 1–12 — already computed, use directly)
- year   (int, e.g. 2024 — already computed, use directly)

CRITICAL TYPE RULE: 'date' is already datetime64.
- NEVER do: df['date'] + ' ' + df['time']  ← this will crash
- ALWAYS filter by year/quarter using the integer columns: df[df['year'] == 2024]
```

Note the **anti-pattern callouts** ("NEVER do...", "ALWAYS use..."). These are critical for avoiding repeat mistakes the model makes when the type system is non-obvious.

### 3. Live-built schema block (M5_UGL_1_R — TinyDB)

For dynamic data, build the schema description from the live data:

```python
schema_block = inv_utils.build_schema_block(inventory_tbl, transactions_tbl)
prompt = PROMPT.format(schema_block=schema_block, question=user_question)
```

`build_schema_block` walks the tables and emits:

```
Inventory Table (inventory_tbl)
- item_id (string): Unique product identifier (e.g., SG001)
- name (string): Style of sunglasses (e.g., Aviator, Round)
- quantity_in_stock (int): Current stock available
- price (float): Price in USD
Sample row: {"item_id":"SG001","name":"Aviator","quantity_in_stock":23,"price":80}
```

Sample rows are gold — the model sees not just types but realistic values.

## Where to put the schema in the prompt

Patterns that work:

- **Right after role**, before constraints: "You are a SQL assistant. Schema: ... User question: ..."
- **In a labeled section** with a clear header: `Database Schema & Samples (read-only):` then the block.
- **Repeat critical rules near the end.** If a column is `datetime64`, mention it at the top *and* in the constraints near the bottom.

> [!info]+ Why "already computed" matters
> A subtle but recurring point: if your dataset has both `date` (datetime64) and a derived `quarter` (int), tell the model which to use. Otherwise it'll write `df['date'].dt.quarter == 1` when `df['quarter'] == 1` is simpler and faster. The schema is the place to communicate this.

## Seen in

- [[M2 UGL 1]] — DataFrame columns + type rules for matplotlib code generation.
- [[M2 UGL 2]] — SQL schema as inline string for SQL generation.
- [[M5 UGL 1 R]] — `build_schema_block(...)` injected into the [[Code as Plan]] prompt with sample rows.

## Related

- [[Code as Plan]] — schema injection is a prerequisite for code-as-plan.
- [[External Feedback Reflection]] — for catching schema-misuse bugs the prompt couldn't prevent.
