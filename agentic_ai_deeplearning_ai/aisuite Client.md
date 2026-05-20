---
tags: [infrastructure, agentic-ai, sdk]
aliases: [aisuite, ai.Client, andrew ng aisuite]
---

# aisuite Client

A unified Python client for calling LLMs from multiple providers (OpenAI, Anthropic, etc.) through a single API surface that mimics OpenAI's `chat.completions.create()`. Built by Andrew Ng's team ([aisuite repo](https://github.com/andrewyng/aisuite)). The default client used throughout the course.

The pitch: write your code once against the OpenAI-shaped API, switch providers by changing a model string. It also adds auto-orchestration for tool calls.

## Initialization

```python
import aisuite as ai
from dotenv import load_dotenv

load_dotenv()           # picks up OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
client = ai.Client()
```

The convention across labs is to expose a module-level `CLIENT` or `client` after `load_dotenv()` puts API keys into the environment.

## Provider-prefixed model strings

The big convention: `<provider>:<model>`. The provider before the colon tells aisuite which backend to route to.

```python
client.chat.completions.create(model="openai:gpt-4o",         ...)
client.chat.completions.create(model="openai:gpt-4.1-mini",   ...)
client.chat.completions.create(model="openai:o4-mini",        ...)
client.chat.completions.create(model="anthropic:claude-sonnet-4-6", ...)
```

Swapping providers becomes a one-character change.

## API shape — OpenAI-compatible

The call shape mirrors `openai.OpenAI().chat.completions.create()` exactly:

```python
response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "What time is it?"},
    ],
    temperature=0,
)
content = response.choices[0].message.content
```

If you know the OpenAI SDK, you know aisuite. Differences are subtle.

## aisuite's superpower — auto tool-call loops

The killer feature: when you pass Python functions as tools and set `max_turns=N`, aisuite:

1. Auto-generates the tool schema from the function's docstring + signature.
2. Loops up to N turns, executing tool calls and feeding results back to the LLM.
3. Returns when the LLM produces a final assistant message.

```python
def get_current_time():
    """Returns the current time as a string."""
    return datetime.now().strftime("%H:%M:%S")

response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": "What time is it?"}],
    tools=[get_current_time],     # ← Python function, not a schema
    max_turns=5,                   # ← auto-loop budget
)
```

See [[Tool Calling - Auto Orchestration]] for the full pattern.

## Inspecting the response (debugging)

`display_functions.pretty_print_chat_completion(response)` in the labs is a thin helper that shows the conversation trace — what tools were called, with what arguments, in what order. Worth replicating in your own code.

For raw inspection:

```python
import json
print(json.dumps(response.model_dump(), indent=2, default=str))
```

> [!tip]+ When aisuite is enough vs. when to drop down
> aisuite is the right tool when:
>
> - You want fast iteration with auto tool-calling.
> - You want provider-portable code.
> - You don't need per-call control over the tool-call loop.
>
> Drop down to the raw OpenAI SDK (or provider-specific SDK) when:
>
> - You need to inspect/transform tool calls before executing — see [[Tool Calling - Manual Loop]].
> - The feature you need isn't supported (e.g., image generation in `M5_UGL_2` — aisuite at lab time doesn't support `images.generate`).
> - You need provider-specific features (Anthropic's prompt caching, OpenAI's structured outputs).

## Hybrid use is fine

`M5_UGL_2` does exactly this:

```python
import aisuite
import openai

client        = aisuite.Client()                 # text chat with auto-tools
openai_client = openai.OpenAI()                   # image generation

# ... text agents use `client.chat.completions.create(...)`
# ... designer agent uses `openai_client.images.generate(...)`
```

No religion required — pick the right SDK per agent.

## Seen in

- Used in essentially every notebook except [[C1M3 Assignment]] and [[M5 UGL 1 R]] (which use raw `openai`) and the image-generation step of [[M5 UGL 2]] (which also uses raw `openai`).

## Related

- [[Tool Calling - Auto Orchestration]] — the main reason to reach for aisuite.
- [[OpenAI Tool Call Protocol]] — what aisuite hides.
- [[Mixed-Model Strategy]] — easier with aisuite's provider abstraction.
