---
tags: [infrastructure, agentic-ai, sdk, openai]
aliases: [tool_calls protocol, function calling wire format]
---

# OpenAI Tool Call Protocol

The wire-level conversation pattern used by OpenAI's `chat.completions.create()` for function calling. When you pass a `tools` schema and the model decides to call one, it returns a message with a `tool_calls` field instead of `content`. You execute the tool locally, append a `role: "tool"` message with the same `tool_call_id`, and re-call.

This is what [[Tool Calling - Auto Orchestration]] (via aisuite) does for you and what [[Tool Calling - Manual Loop]] makes you do by hand.

## The tool schema

The schema is a JSON object describing the function's name, description, and parameters (JSON Schema):

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Returns the current time as a string.",
        "parameters": {}
    }
}, {
    "type": "function",
    "function": {
        "name": "arxiv_search_tool",
        "description": "Search arXiv for academic papers.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
}]
```

Pass this to the LLM via `tools=tools`.

## The two outcomes

After calling `chat.completions.create(...)`, the response message has one of two shapes:

**Final answer:**
```python
message.content      = "The time is 14:23:15."
message.tool_calls   = None
```

**Tool request:**
```python
message.content     = None
message.tool_calls  = [
    ChatCompletionMessageFunctionToolCall(
        id='call_ymMki5TBB91efJhMPjgoqjop',
        type='function',
        function=Function(
            name='get_current_time',
            arguments='{}',   # JSON string!
        ),
    ),
]
```

The `arguments` field is a **JSON-encoded string**, not a dict. Parse with `json.loads`.

## Tool result message shape

After running the tool locally, append a `role: "tool"` message:

```python
messages.append(response.choices[0].message)    # the assistant's tool-call message
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,    # ← MUST match the assistant's call ID
    "name": tool_call.function.name,  # optional but conventional
    "content": json.dumps(result),    # ← stringified result
})
```

The `tool_call_id` is what ties the result to the request. Don't skip it. If the assistant requested multiple parallel tool calls, append one result message per call.

## Multiple tool calls in one turn

The model can request several tools in a single message. The `tool_calls` list will have multiple entries. Execute each and append each result:

```python
for tool_call in msg.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    result = TOOL_MAPPING[name](**args)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": name,
        "content": json.dumps(result),
    })
```

## `tool_choice` control

The `tool_choice` parameter controls whether and how the model can use tools:

| Value                                | Meaning                                      |
| ------------------------------------ | -------------------------------------------- |
| `"auto"` (default)                   | Model decides whether and which tool to call |
| `"none"`                             | Disable tool use this turn                   |
| `"required"`                         | Force the model to call some tool            |
| `{"type": "function", "function": {"name": "X"}}` | Force tool `X`                       |

The course consistently uses `tool_choice="auto"`.

## Inspecting tool-call output

```python
ChatCompletionMessage(
    content=None,
    role='assistant',
    tool_calls=[
        ChatCompletionMessageFunctionToolCall(
            id='call_ymMki5TBB91efJhMPjgoqjop',
            function=Function(
                arguments='{"query":"radio observations of recurrent novae","max_results":5}',
                name='arxiv_search_tool',
            ),
            type='function'
        )
    ]
)
```

Always:

```python
call_name = call.function.name         # str
call_args = json.loads(call.function.arguments)   # dict
```

## Termination

The loop ends when the assistant returns a message with `content` set and no `tool_calls`. Plus a defensive max-turns cap:

```python
for _ in range(max_turns):
    response = client.chat.completions.create(...)
    msg = response.choices[0].message
    messages.append(msg)
    if not msg.tool_calls:
        return msg.content
    # ... execute and append tool results
```

> [!failure]+ Common pitfalls
> - **Forgetting to append the assistant's tool-call message** before the tool result. The OpenAI API requires both — the request *and* the result must be in the conversation.
> - **Wrong `tool_call_id`.** Must exactly match the assistant's; otherwise the API errors out or the model gets confused.
> - **Tool result not JSON-stringified.** `content` must be a string. Wrap dicts/lists with `json.dumps`.
> - **No `max_turns` cap.** Without one you can infinite-loop on stubborn tool requests.

## Seen in

- [[C1M3 Assignment]] — the canonical implementation of this protocol.
- [[M3 UGL 1]] (later cells) — manual schema + manual handling shown explicitly.

## Related

- [[Tool Calling - Manual Loop]] — the application pattern.
- [[Tool Calling - Auto Orchestration]] — aisuite hides this protocol.
- [[Agent Registry and Routing]] — `TOOL_MAPPING` for dispatch.
