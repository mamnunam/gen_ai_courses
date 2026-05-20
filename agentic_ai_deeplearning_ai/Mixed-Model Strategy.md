---
tags: [technique, agentic-ai, cost-optimization]
aliases: [model routing, cheap-gen strong-critic, model mixing]
---

# Mixed-Model Strategy

Using different LLMs for different steps in a pipeline based on what each step needs. A cheap, fast model for high-volume generation; a stronger reasoning model for critique or planning where quality matters more than latency. This is the lowest-effort cost optimization in agentic workflows.

## The recipe

| Step             | Model                                  | Why                                                 |
| ---------------- | -------------------------------------- | --------------------------------------------------- |
| Generation (V1)  | `gpt-4o-mini`, `gpt-4.1-mini`, `gpt-3.5-turbo` | Fast, cheap, "good enough" for a first draft        |
| Critique         | `o4-mini`, `gpt-4.1`, `claude-sonnet-4-6` | Reasoning quality matters; called less often        |
| Revision (V2)    | Either, depending on cost vs. quality  | Bigger model if critique is hard to operationalize  |
| Tool-calling loop | `gpt-4o`, `gpt-4.1`                   | Function-calling reliability is the priority        |
| Image generation | `gpt-image-1-mini`                    | Dedicated image model (not general chat)            |

## The lab-suggested split

`M2_UGL_1` makes it explicit:

```python
generation_model = "gpt-4o-mini"     # cheap for V1
reflection_model = "o4-mini"          # stronger reasoning for critique
# Or:
# reflection_model = "claude-sonnet-4-6"

_ = run_workflow(
    dataset_path="coffee_sales.csv",
    user_instructions=user_instructions,
    generation_model=generation_model,
    reflection_model=reflection_model,
    image_basename=image_basename,
)
```

`C1M2` (the graded reflection lab) does the same by default:

```python
def generate_draft(topic, model="openai:gpt-4o"):       # creative gen
def reflect_on_draft(draft, model="openai:o4-mini"):    # reasoning critic
def revise_draft(original_draft, reflection,
                 model="openai:gpt-4o"):                # back to creative
```

> [!info]+ Why this works
> - **Generation tasks** are mostly about fluency. Smaller models are surprisingly good at "write me an essay" or "draft this SQL."
> - **Critique tasks** require *reasoning* about whether the output actually satisfies the request — a job o-series and large models do measurably better.
> - **The critic is called once per draft**, while generation can be called many times. Putting the expensive model on the lower-volume side keeps cost down.

> [!warning]+ Anti-pattern: same model everywhere
> Defaulting every step to `gpt-4o` (or `gpt-4.1`) works, but it overpays for steps that don't need that much capability. The downside of generation_model=gpt-4o-mini is occasionally weaker V1; the [[Reflection Pattern]] absorbs that — the critique step catches issues that V1 missed.

## Mixed providers

The labs also illustrate provider mixing — same workflow, different vendors per step:

```python
# Inside reflect_on_image_and_regenerate
lower = model_name.lower()
if "claude" in lower or "anthropic" in lower:
    content = utils.image_anthropic_call(model_name, prompt, media_type, b64)
else:
    content = utils.image_openai_call(model_name, prompt, media_type, b64)
```

aisuite normalizes most of this; for vision and image gen you sometimes still need provider-specific code (see [[Multimodal Image Input]]).

## Quick model menu (course mentions)

- **`openai:gpt-4o`** — well-rounded, fast, default for reasoning + speed.
- **`openai:gpt-4.1`** — strongest general reasoning in the course examples.
- **`openai:gpt-4.1-mini`** — lighter, faster, cheaper.
- **`openai:o4-mini`** — strong reasoning; typical critic choice.
- **`openai:gpt-3.5-turbo`** — fastest/cheapest for simple iteration.
- **`openai:gpt-image-1-mini`** — image generation only.

> [!tip]+ Practical tips
> - Make `generation_model` and `reflection_model` (etc.) function parameters with sensible defaults. Easy to swap during experimentation.
> - Track which model produced which artifact when debugging.
> - Cost isn't the only axis — `o4-mini` (reasoning) is slower than `gpt-4o-mini` (chat) per token. For latency-sensitive paths, that matters.

## Seen in

- [[M2 UGL 1]] — `gpt-4o-mini` for chart code, `o4-mini` for image reflection.
- [[M2 UGL 2]] — `gpt-4.1` for both gen and eval (with note that it tends to give best results for self-reflection).
- [[C1M2 Assignment]] — defaults of `gpt-4o` for draft/revise, `o4-mini` for reflect.
- [[M5 UGL 2]] — `o4-mini` for the agents, `gpt-image-1-mini` for image generation.
