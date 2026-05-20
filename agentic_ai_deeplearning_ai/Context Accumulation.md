---
tags: [concept, agentic-ai, multi-agent, state]
aliases: [history accumulation, agent state, shared context]
---

# Context Accumulation

In multi-step or multi-agent workflows, each step needs to see what previous steps produced. The cheap and explicit way to do this: **keep a `history` list** and inject prior outputs into each new agent call's prompt.

No state machines, no frameworks, no shared object stores — just a list and string concatenation.

## The C1M5 pattern

The executor builds context from prior steps each iteration and injects it into the next agent's task:

```python
history = []   # list of (step_description, agent_name, output) tuples

for step in plan_steps:
    # Router decides which agent + what task (see [[Agent Registry and Routing]])
    agent_info = json.loads(clean_json_block(router_response))
    agent_name, task = agent_info["agent"], agent_info["task"]

    # Build context from prior history
    context = "\n".join([
        f"Step {j+1} executed by {a}:\n{r}"
        for j, (s, a, r) in enumerate(history)
    ])
    enriched_task = f"""
    You are {agent_name}.

    Here is the context of what has been done so far:
    {context}

    Your next task is:
    {task}
    """

    output = agent_registry[agent_name](enriched_task)
    history.append((step, agent_name, output))
```

Each agent sees the full prior history before being asked to do its part.

## Variants

### Append to a messages list (intra-agent)

Within a single agent's tool-calling loop, context accumulation is automatic — every tool result is appended to `messages`, and the next LLM call sees the entire conversation:

```python
messages.append(response.choices[0].message)             # the LLM's response
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": str(tool_result),
})
# Next call sees the full history
response2 = client.chat.completions.create(messages=messages, ...)
```

This is the same pattern at a smaller scale — the conversation IS the history.

### Direct hand-off (sequential pipeline)

In a static [[Sequential Multi-Agent Pipeline]] you don't need a history list; you pass outputs as explicit arguments:

```python
trend_summary = market_research_agent()
visual        = graphic_designer_agent(trend_insights=trend_summary)
quote_result  = copywriter_agent(image_path=visual["image_path"],
                                 trend_summary=trend_summary)
```

Same accumulation, but the pipeline structure documents the dependency graph.

### Stateful agent object

You could wrap the history list inside an Agent class with methods. The labs don't — they keep history as a top-level list. Simpler, easier to inspect during debugging.

## What to put in the context

- **Full prior outputs** for short chains (~4 steps). Simple, complete information.
- **Summarized outputs** for long chains — call an LLM to summarize step N's output before passing to step N+1, to keep tokens bounded.
- **Last N outputs only** for chains where context windows are tight; sacrifices some coherence for cost.

The C1M5 lab caps plans at 4 steps specifically to keep the running history small.

> [!failure]+ When this breaks
> - **Context overflow.** With long histories, you hit token limits. Mitigation: summarize older entries or use retrieval (RAG) over the history.
> - **Garbage in, garbage out.** If an earlier step produces low-quality output, every later step is poisoned by it. Mitigation: add a [[Reflection Pattern]] step before passing forward.
> - **Lost in the middle.** LLMs sometimes underweight middle entries in long histories. Mitigation: re-emphasize critical prior outputs in the immediate prompt.

> [!info]+ Why a list works
> The course's design philosophy is to **build patterns from primitives**. Context accumulation could be a fancy state graph (LangGraph), but a Python list with string concatenation is:
>
> - Inspectable (`print(history)` shows everything).
> - Easy to test (just compare lists).
> - Trivially serializable (`json.dumps(history)`).
> - One file shorter than any framework alternative.
>
> Reach for fancier state management only when this approach actually breaks.

## Seen in

- [[C1M5 Assignment]] — `history` list in `executor_agent`, with context-building string interpolation.
- [[C1M3 Assignment]] — messages-list accumulation in the manual tool-call loop.
- [[M5 UGL 2]] — implicit accumulation via sequential function arguments.

## Related

- [[Planner-Executor Architecture]] — the larger pattern.
- [[Sequential Multi-Agent Pipeline]] — explicit, static accumulation via function args.
- [[Tool Calling - Manual Loop]] — within-agent accumulation in `messages`.
