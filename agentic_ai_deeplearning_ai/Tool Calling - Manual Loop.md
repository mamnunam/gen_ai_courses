---
tags: [pattern, agentic-ai, tools, openai-sdk]
aliases: [manual tool loop, tool_calls protocol]
---

# Tool Calling — Manual Loop

The explicit version of tool calling: you write the tool schema yourself, run the LLM, inspect the `tool_calls` field on the response message, execute the tool locally, append a `role: "tool"` message, and re-call the LLM. Repeat until the model returns a final assistant message.

This is what [[Tool Calling - Auto Orchestration]] hides. The course's graded labs use this form because they're built on the raw `openai` SDK rather than `aisuite`.

> [!tip]+ When to use
> - Raw OpenAI SDK (no aisuite wrapper).
> - You want to log, transform, or veto individual tool calls.
> - You want the tool list to be data, not Python functions (e.g., dispatched via a name → function dict).
> - Educational clarity — seeing exactly what aisuite does for you.

## Manual tool schema

Without aisuite's docstring inference, you write the tool schema by hand:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Returns the current time as a string.",
        "parameters": {}   # no args for this tool
    }
}]
```

For tools with parameters, you spell out the JSON Schema for `parameters` including types and required fields.

## The loop (graded-lab style)

The C1M3 assignment is the canonical form. It uses a tool-name → function dict and runs a bounded for-loop:

```python
TOOL_MAPPING = {
    "tavily_search_tool": research_tools.tavily_search_tool,
    "arxiv_search_tool":  research_tools.arxiv_search_tool,
}

def generate_research_report_with_tools(prompt, model="gpt-4o"):
    messages = [
        {"role": "system", "content": "You are a research assistant..."},
        {"role": "user",   "content": prompt},
    ]
    tools = [research_tools.arxiv_tool_def, research_tools.tavily_tool_def]

    for _ in range(10):   # max_turns budget
        response = CLIENT.chat.completions.create(
            model=model, messages=messages,
            tools=tools, tool_choice="auto", temperature=1,
        )
        msg = response.choices[0].message
        messages.append(msg)

        # No tool calls → final answer
        if not msg.tool_calls:
            return msg.content

        # Execute each tool call and append the result
        for call in msg.tool_calls:
            tool_name = call.function.name
            args = json.loads(call.function.arguments)
            try:
                tool_func = TOOL_MAPPING[tool_name]
                result = tool_func(**args)
            except Exception as e:
                result = {"error": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,     # must match the LLM's call ID
                "name": tool_name,
                "content": json.dumps(result),
            })
```

## The protocol in plain language

1. Send the user message + tools list to the LLM.
2. LLM responds with either:
   - **Final answer** → `msg.content` is set, `msg.tool_calls` is empty. Done.
   - **Tool request** → `msg.tool_calls` is a list of `ChatCompletionMessageFunctionToolCall` objects, each with `id`, `function.name`, `function.arguments` (JSON string).
3. For each tool call: parse args, run the function, append a `role: "tool"` message with the **same `tool_call_id`** and the JSON-stringified result.
4. Loop. The LLM now sees the conversation including the tool results.

The `tool_call_id` is the connective tissue — it tells the LLM which prior tool call this result corresponds to.

## The `ChatCompletionMessage` shape

For reference (from the C1M3 hints):

```python
ChatCompletionMessage(
    content=None,
    role='assistant',
    tool_calls=[
        ChatCompletionMessageFunctionToolCall(
            id='call_ymMki5TBB91efJhMPjgoqjop',
            function=Function(
                arguments='{"query":"radio observations of recurrent novae","max_results":5}',
                name='arxiv_search_tool'
            ),
            type='function'
        )
    ]
)
```

So `call.function.name` and `call.function.arguments` (a JSON string you parse) are what you need.

## Variant: while-True with explicit termination

`M5_UGL_2`'s `market_research_agent` uses `while True` instead of a bounded for-loop — it relies on the LLM returning `msg.content` to break:

```python
while True:
    response = client.chat.completions.create(...)
    msg = response.choices[0].message
    if msg.content:                       # final answer reached
        return msg.content
    if msg.tool_calls:
        for tc in msg.tool_calls:
            result = tools.handle_tool_call(tc)
            messages.append(msg)
            messages.append(tools.create_tool_response_message(tc, result))
    else:
        return "[⚠️ Unexpected: no content or tool_calls]"
```

In practice, **prefer a bounded loop** — the for-loop with `range(max_turns)` cannot infinite-loop. `while True` requires confidence that the model will terminate.

## Seen in

- [[C1M3 Assignment]] — the canonical manual loop with `TOOL_MAPPING` and bounded turns.
- [[M3 UGL 1]] (later cells) — manual schema + manual handling shown side-by-side with the auto version.
- [[M5 UGL 2]] — the Market Research Agent uses a `while True` variant around an `aisuite` call (a hybrid).

## Related

- [[Tool Calling - Auto Orchestration]] — the higher-level version.
- [[OpenAI Tool Call Protocol]] — the protocol details abstracted.
- [[Structured JSON Outputs]] — a different way to extract structured info without tools.
