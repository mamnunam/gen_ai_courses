---
tags: [pattern, agentic-ai, tools, aisuite]
aliases: [auto tool calling, aisuite tools, max_turns]
---

# Tool Calling — Auto Orchestration

The pattern where you hand the LLM a list of Python functions plus a `max_turns` budget, and let the SDK (here, `aisuite`) handle the whole tool-call → execute → return loop for you. You write Python functions with docstrings; the LLM does the rest.

This is the path of least resistance for adding tools to an agent. Contrast with [[Tool Calling - Manual Loop]], where you handle the protocol yourself.

## The mechanic

`aisuite` does three things automatically:

1. Inspects your function signature + docstring and turns it into a JSON tool schema for the LLM (no manual schema needed).
2. When the LLM responds with a `tool_calls` message, it parses the arguments, calls the local Python function, and appends the result as a `role: "tool"` message.
3. Repeats up to `max_turns` until the LLM returns a final assistant message.

## Minimal example

```python
from datetime import datetime
import aisuite as ai

client = ai.Client()

def get_current_time():
    """Returns the current time as a string."""
    return datetime.now().strftime("%H:%M:%S")

response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "What time is it?"}],
    tools=[get_current_time],   # ← just pass the function
    max_turns=5                  # ← budget for the auto-loop
)
print(response.choices[0].message.content)
```

That's it. `aisuite` synthesizes the schema from the docstring, runs `get_current_time()` locally when the LLM asks for it, and returns the final natural-language answer.

## Multi-tool

Same call, more tools. The LLM picks based on the prompt's intent:

```python
response = client.chat.completions.create(
    model="openai:o4-mini",
    messages=[{"role": "user", "content":
        "Make a QR code that goes to www.deeplearning.com from dl_logo.jpg. "
        "Also write me a txt note with the current weather."
    }],
    tools=[
        get_weather_from_ip,
        get_current_time,
        write_txt_file,
        generate_qr_code,
    ],
    max_turns=10,
)
```

The LLM will chain tools in the order that satisfies data dependencies (call `get_weather_from_ip` *before* `write_txt_file` even if the user asked for the note "first"), then assemble the final response in the order the user requested.

## With explicit `tool_choice`

You can force or allow tool selection. In `M4_UGL_1` and the C1M5 assignment, the pattern is:

```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=[arxiv_search_tool, tavily_search_tool, wikipedia_search_tool],
    tool_choice="auto",   # let the model decide whether/which tool to call
    max_turns=5,
)
```

`tool_choice="auto"` is the default for `aisuite` but it's worth stating explicitly when reading later.

## Docstring conventions

For auto-schema to work well:

- One-line summary first (this becomes the tool description).
- Args section with each parameter and a short description (these become parameter descriptions).
- Be specific about formats, units, and what the tool does *not* do.

```python
def write_txt_file(file_path: str, content: str):
    """
    Write a string into a .txt file (overwrites if exists).
    Args:
        file_path (str): Destination path.
        content (str): Text to write.
    Returns:
        str: Path to the written file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return file_path
```

> [!tip]+ When to reach for manual instead
> - You need to inspect or transform tool-call arguments before executing.
> - You need to gate tool execution behind a confirmation step (human-in-the-loop).
> - You're using the raw OpenAI SDK without aisuite (graded labs do this).
> - You need to handle errors per-call and inject them back as tool responses.
>
> For any of those, use [[Tool Calling - Manual Loop]].

## Seen in

- [[M3 UGL 1]] — the canonical introduction (time, weather, file-write, QR-code tools).
- [[M3 UGL 2]] — multi-step email assistant (search → mark read → reply).
- [[M4 UGL 1]] — `find_references()` with `tool_choice="auto"` over arxiv/tavily/wikipedia.
- [[C1M5 Assignment]] — `research_agent` with three search tools, `max_turns=6`.
- [[M5 UGL 2]] — Market Research Agent uses a manual `while True` loop *around* aisuite calls (a hybrid).

## Related

- [[Tool Calling - Manual Loop]] — what aisuite hides.
- [[Capability Boundary]] — the agent can do exactly what the tool list lets it do.
- [[aisuite Client]] — the unified-provider client.
- [[OpenAI Tool Call Protocol]] — the wire protocol underneath.
