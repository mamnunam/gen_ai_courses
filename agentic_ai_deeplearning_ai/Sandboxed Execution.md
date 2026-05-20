---
tags: [technique, agentic-ai, safety, code-execution]
aliases: [exec sandbox, controlled namespace]
---

# Sandboxed Execution

Running LLM-generated code in a **controlled namespace** — a dict of pre-bound globals and locals you pass to Python's `exec()`. The model sees only the variables you put there, not the host process's full scope. You also capture stdout and exceptions so you can audit what the code did.

It's not real sandboxing (the code can still import `os`, write files, hit the network) — it's a *namespace* sandbox. For untrusted code you'd add seccomp, containers, or a separate process. For trusted-but-non-deterministic LLM output it's the right level of friction.

## The two ingredients

1. A pre-bound namespace — what the code can use.
2. A way to read back what it set — what the code produced.

## Pattern A — minimal (M2_UGL_1)

For executing chart-generation code:

```python
import re

# Extract code from <execute_python> tags (see [[Tag-Based Code Extraction]])
match = re.search(r"<execute_python>([\s\S]*?)</execute_python>", code_v1)
if match:
    initial_code = match.group(1).strip()
    exec_globals = {"df": df}        # ← pre-bound: the DataFrame
    exec(initial_code, exec_globals)
```

`df` is the only thing the code can see by default. (It can still import matplotlib, etc., because Python's import system is in scope.) Side effects (saved chart files) happen in the host filesystem.

## Pattern B — full executor with stdout capture (M5_UGL_1_R)

The customer-service agent uses a more careful executor:

```python
import io, sys, traceback
from typing import Any, Dict, Optional

def execute_generated_code(
    code_or_content: str, *,
    db, inventory_tbl, transactions_tbl,
    user_request: Optional[str] = None,
) -> Dict[str, Any]:
    code = _extract_execute_block(code_or_content)   # tag extraction

    # 1) Pre-bound namespace: only what the plan needs
    SAFE_GLOBALS = {
        "Query": Query,
        "get_current_balance": inv_utils.get_current_balance,
        "next_transaction_id": inv_utils.next_transaction_id,
        "user_request": user_request or "",
    }
    SAFE_LOCALS = {
        "db": db,
        "inventory_tbl": inventory_tbl,
        "transactions_tbl": transactions_tbl,
    }

    # 2) Capture stdout and exceptions
    _stdout_buf, _old_stdout = io.StringIO(), sys.stdout
    sys.stdout = _stdout_buf
    err_text = None
    try:
        exec(code, SAFE_GLOBALS, SAFE_LOCALS)
    except Exception:
        err_text = traceback.format_exc()
    finally:
        sys.stdout = _old_stdout

    printed = _stdout_buf.getvalue().strip()

    # 3) Read back the convention answer variables
    answer = (
        SAFE_LOCALS.get("answer_text")
        or SAFE_LOCALS.get("answer_rows")
        or SAFE_LOCALS.get("answer_json")
    )

    return {
        "code": code,
        "stdout": printed,
        "error": err_text,
        "answer": answer,
        "transactions_tbl": transactions_tbl.all(),  # before/after snapshot
        "inventory_tbl": inventory_tbl.all(),
    }
```

Three additions vs. the minimal pattern:

- **Stdout capture.** Replace `sys.stdout` with a `StringIO` so prints land in `printed` instead of the notebook.
- **Exception capture.** `traceback.format_exc()` so the LLM (in a follow-up reflection step) can see what went wrong.
- **Convention read-back.** The prompt says "set `answer_text`"; the executor reads that variable out of `SAFE_LOCALS` after `exec()`.

## Convention variables — the plan→executor handoff

The pattern uses three reserved names:

| Variable      | Purpose                                          |
| ------------- | ------------------------------------------------ |
| `answer_text` | Short, customer-facing string (required)         |
| `answer_rows` | Optional structured rows                          |
| `answer_json` | Optional JSON-serializable result                 |

This makes the contract explicit: the executor knows where to look for the result, and the model knows where to put it.

## Before/after snapshots

For mutating workflows (the `M5_UGL_1_R` "return 2 Aviators" example), capturing the full table state before and after execution is essential for auditing:

```python
utils.print_html(json.dumps(inventory_tbl.all(), indent=2), title="Inventory · Before")
result = execute_generated_code(...)
utils.print_html(json.dumps(inventory_tbl.all(), indent=2), title="Inventory · After")
```

This turns opaque "LLM mutated state" into a diff you can review.

> [!warning]+ Limits and risks
> - This is *namespace* sandboxing, not OS-level sandboxing. The code can still `import os` and `os.remove(...)`.
> - For real isolation: run the exec in a subprocess with restricted permissions, a Docker container, or a Pyodide/seccomp environment.
> - Even with namespace sandboxing, infinite loops will hang the host. Add a timeout via `signal.alarm()` or run in a thread with a deadline.

## Seen in

- [[M2 UGL 1]] — minimal `exec(code, {"df": df})` for chart generation.
- [[M5 UGL 1 R]] — full executor with stdout capture, error capture, and convention read-back.

## Related

- [[Tag-Based Code Extraction]] — getting the code out of the LLM response first.
- [[Code as Plan]] — the larger pattern that needs this executor.
