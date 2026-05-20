---
tags: [pattern, agentic-ai, multi-agent, planning]
aliases: [planner-executor, plan and execute, orchestrator pattern]
---

# Planner-Executor Architecture

A multi-agent pattern where one LLM call produces a **plan** as a list of natural-language steps, and a second loop **executes** each step by routing it to the right specialist agent. The executor maintains shared history so each step has context from the previous ones.

Contrasts with [[Code as Plan]]: there, the plan *is* Python. Here, the plan is structured English and an LLM-based router decides which specialist handles each line.

## Three roles

1. **Planner** — produces an ordered list of steps for a research/writing/editing task.
2. **Executor (orchestrator)** — for each step, asks an LLM "which specialist should run this?" then dispatches.
3. **Specialists** — single-purpose agents (research, writer, editor) differentiated only by system prompt.

## Planner

The planner emits a Python list of strings, parsed with `ast.literal_eval`:

```python
def planner_agent(topic, model="openai:o4-mini") -> list[str]:
    user_prompt = f"""
    You are a planning agent organizing a multi-agent research workflow.

    🧠 Available agents:
    - research agent (web, Wikipedia, arXiv search)
    - writer agent (draft summaries)
    - editor agent (reflect/revise)

    🎯 Write a clear, step-by-step research plan as a valid Python list,
       where each step is a string. Each step must be atomic and executable
       and use only the above agents.

    ✅ Final step should be a Markdown report containing the complete research.
    ✅ Return ONLY the Python list, no explanations.

    Topic: "{topic}"
    """
    response = CLIENT.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=1,
    )
    return ast.literal_eval(response.choices[0].message.content.strip())
```

The plan looks like:

```python
[
  "Search Wikipedia and arXiv for the ensemble Kalman filter to gather core concepts.",
  "Summarize the key papers and concepts into a research draft.",
  "Reflect on and revise the draft for clarity.",
  "Produce a final Markdown report with the polished content.",
]
```

## Agent registry (the routing table)

A plain dict maps agent names → callables. See [[Agent Registry and Routing]] for the general pattern.

```python
agent_registry = {
    "research_agent": research_agent,   # uses tools (arxiv/tavily/wikipedia)
    "writer_agent":   writer_agent,
    "editor_agent":   editor_agent,
}
```

## Executor with per-step routing

For each step, a small LLM call decides which agent runs it. This is the key flexibility: the routing is dynamic, not encoded in the plan.

```python
def executor_agent(topic, model="openai:gpt-4o", limit_steps=True):
    plan_steps = planner_agent(topic)
    if limit_steps:
        plan_steps = plan_steps[:4]            # cap for cost/time

    history = []
    for step in plan_steps:
        # 1) Decide which agent gets this step + extract the task
        agent_decision_prompt = f"""
        You are an execution manager. Given the instruction, decide which agent
        should perform it and extract the clean task.

        Return ONLY a JSON object with keys:
        - "agent": one of ["research_agent","editor_agent","writer_agent"]
        - "task": instruction string for the agent

        Instruction: "{step}"
        """
        raw = CLIENT.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": agent_decision_prompt}],
            temperature=0,
        ).choices[0].message.content
        agent_info = json.loads(clean_json_block(raw))   # see helper below
        agent_name, task = agent_info["agent"], agent_info["task"]

        # 2) Build context from prior steps' outputs
        context = "\n".join(
            f"Step {j+1} executed by {a}:\n{r}"
            for j, (_, a, r) in enumerate(history)
        )
        enriched_task = f"""
        You are {agent_name}.

        Here is the context of what has been done so far:
        {context}

        Your next task is:
        {task}
        """

        # 3) Dispatch
        if agent_name in agent_registry:
            output = agent_registry[agent_name](enriched_task)
        else:
            output = f"⚠️ Unknown agent: {agent_name}"

        history.append((step, agent_name, output))

    return history
```

`clean_json_block` strips markdown code fences from the LLM's JSON response — a recurring need:

```python
def clean_json_block(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()
```

## Specialists as system-prompt-only agents

The writer and editor agents are literally the same function shape, differentiated only by their system prompt:

```python
def writer_agent(task, model="openai:gpt-4o"):
    system_prompt = "You are a writing agent focused on academic or technical content."
    return CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": task},
        ],
        temperature=1.0,
    ).choices[0].message.content
```

The research agent additionally has tools (see [[Tool Calling - Auto Orchestration]]).

> [!warning]+ Trade-offs
> - **Strength:** dynamic routing — the plan doesn't need to know which agent handles which step. New specialists slot in via the registry.
> - **Cost:** every step is at least 2 LLM calls (router + specialist), plus tool calls inside the research agent. Cap `max_steps` (the lab uses 4) to keep runs bounded.
> - **Fragility:** the planner's output must be a parseable Python list; the router's must be valid JSON. Both are protected by `ast.literal_eval` + `clean_json_block` but break occasionally.

## Variations

- Static routing: planner emits `(agent_name, task)` tuples directly — saves the router call but loses flexibility.
- LangGraph/CrewAI: similar idea, formalized into a state machine. The course doesn't use these — it builds the pattern from primitives.

## Seen in

- [[C1M5 Assignment]] — the canonical implementation: planner + research/writer/editor + executor with dynamic routing.

## Related

- [[Sequential Multi-Agent Pipeline]] — the simpler, static-DAG cousin.
- [[Agent Registry and Routing]] — the dispatch mechanism.
- [[Context Accumulation]] — passing prior step outputs forward.
- [[Code as Plan]] — the alternative where the plan is code.
