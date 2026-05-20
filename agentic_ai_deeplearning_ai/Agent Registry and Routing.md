---
tags: [technique, agentic-ai, multi-agent, routing]
aliases: [agent dispatch, agent_registry, LLM-as-router]
---

# Agent Registry and Routing

A dict that maps **agent names (strings)** to **callable agent functions**. Combined with a small LLM call that returns the agent name to dispatch to, this gives you dynamic routing in a multi-agent system. It's the simplest possible "agent dispatch" — a plain Python dict plus `json.loads`.

The same pattern shows up for tool mapping (see [[Tool Calling - Manual Loop]]) — strings in the LLM's output get translated to callables via a lookup table.

## The dict

```python
agent_registry = {
    "research_agent": research_agent,
    "editor_agent":   editor_agent,
    "writer_agent":   writer_agent,
}
```

Each value is a Python function with a uniform signature (typically takes a `task: str`, returns a `str`). The dict is the routing table.

## The router LLM call

A small, focused LLM call per step decides which agent should handle it. It returns JSON with the agent name + extracted task:

```python
agent_decision_prompt = f"""
You are an execution manager for a multi-agent research team.

Given the following instruction, identify which agent should perform it
and extract the clean task.

Return only a valid JSON object with two keys:
- "agent": one of ["research_agent", "editor_agent", "writer_agent"]
- "task": a string with the instruction that the agent should follow

Only respond with a valid JSON object. Do not include explanations or markdown.

Instruction: "{step}"
"""
response = CLIENT.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": agent_decision_prompt}],
    temperature=0,    # deterministic — we want consistent routing
)
raw_content   = response.choices[0].message.content
cleaned_json  = clean_json_block(raw_content)
agent_info    = json.loads(cleaned_json)
agent_name    = agent_info["agent"]
task          = agent_info["task"]
```

Key design choices:

- **`temperature=0`** — routing should be consistent; don't let the model freelance.
- **Explicit allowlist in the prompt** — `one of ["research_agent", "editor_agent", "writer_agent"]`. This constrains the output space.
- **Returns two fields** — the agent and the *cleaned* task. The router can rewrite the instruction to be more agent-appropriate.

## Dispatch

Once you have the name, dispatch through the registry:

```python
if agent_name in agent_registry:
    output = agent_registry[agent_name](enriched_task)
else:
    output = f"⚠️ Unknown agent: {agent_name}"
history.append((step, agent_name, output))
```

The defensive branch matters — even with the prompt's allowlist, the model can hallucinate. Handle the unknown case explicitly.

> [!info]+ Why a dict and not class inheritance
> The lab keeps each "agent" as a plain function differentiated by system prompt, then routes via a dict. Compared to a class-hierarchy approach:
>
> - **Less ceremony.** Adding an agent means writing a function and adding one line to the dict.
> - **Easier to test.** Each function is independent; no shared state.
> - **Composable.** You can wrap an agent (e.g., add caching, logging) without subclassing.
>
> If you need shared infrastructure (retries, logging, telemetry), wrap functions with decorators rather than introducing a base class.

## Same pattern: tool mapping

The C1M3 graded lab uses an identical structure for tools instead of agents:

```python
TOOL_MAPPING = {
    "tavily_search_tool": research_tools.tavily_search_tool,
    "arxiv_search_tool":  research_tools.arxiv_search_tool,
}

# Later in the manual tool-call loop:
tool_func = TOOL_MAPPING[tool_name]
result = tool_func(**args)
```

The dispatch is the same — strings (from LLM output) translated to callables via dict lookup.

## Helper: `clean_json_block`

LLMs frequently wrap JSON in markdown fences. This helper strips them:

```python
def clean_json_block(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()
```

Always run this before `json.loads()` on LLM output if you've seen it produce fenced JSON.

## Extensions to consider

- **Per-step fallback agent.** If routing fails or returns an unknown name, hand the step to a `generalist_agent` instead of erroring.
- **Multi-agent dispatch in one step.** Return `agents: [...]` and run them in parallel; collect outputs.
- **Confidence threshold.** Have the router emit a confidence score; below a threshold, escalate to a human or stronger model.

## Seen in

- [[C1M5 Assignment]] — `agent_registry` + per-step LLM router (canonical).
- [[C1M3 Assignment]] — `TOOL_MAPPING` for tool dispatch in the manual loop (same idea, different object).

## Related

- [[Planner-Executor Architecture]] — the larger pattern this enables.
- [[Structured JSON Outputs]] — the router's contract.
- [[Tool Calling - Manual Loop]] — same dict-dispatch idea for tools.
