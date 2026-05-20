---
tags: [pattern, agentic-ai, multi-agent]
aliases: [agent handoff, static multi-agent pipeline]
---

# Sequential Multi-Agent Pipeline

A static DAG of specialized agents, each producing output that the next consumes. No router, no dynamic dispatch — the orchestration is hard-coded in a top-level function that calls agents in order.

This is the cousin of [[Planner-Executor Architecture]] for cases where you already know the steps. Use it when the workflow is fixed and you want maximum transparency.

## The canonical example: marketing campaign

`M5_UGL_2` builds a sunglasses-campaign pipeline with four roles:

```
Market Research Agent  →  Graphic Designer Agent  →  Copywriter Agent  →  Packaging Agent
       (text)                (image + caption)        (quote + image)       (markdown)
```

Each agent has a single specialization and a single input/output contract.

## Top-level orchestration

The whole pipeline is just sequential function calls — no planner, no router:

```python
def run_sunglasses_campaign_pipeline(output_path="campaign_summary.md"):
    # 1. Market research (text)
    trend_summary = market_research_agent()

    # 2. Generate image + caption (multimodal output)
    visual = graphic_designer_agent(trend_insights=trend_summary)
    image_path = visual["image_path"]

    # 3. Generate campaign quote based on image + trends (multimodal input)
    quote_result = copywriter_agent(image_path=image_path,
                                    trend_summary=trend_summary)

    # 4. Package everything into a Markdown report
    md_path = packaging_agent(
        trend_summary=trend_summary,
        image_url=image_path,
        quote=quote_result["quote"],
        justification=quote_result["justification"],
        output_path=output_path,
    )
    return {"trend_summary": trend_summary, "visual": visual,
            "quote": quote_result, "markdown_path": md_path}
```

## Agent-internal shapes

The agents themselves vary in mechanism:

- **Market Research Agent** — manual `while True` tool-calling loop around aisuite calls; uses `tavily_search_tool` + `product_catalog_tool`.
- **Graphic Designer Agent** — two-stage: aisuite text call for prompt+caption JSON, then raw `openai.images.generate(model="gpt-image-1-mini")` for the image. Bridges a framework gap (aisuite doesn't do image generation yet).
- **Copywriter Agent** — multimodal input (image as base64 `image_url` block + trend summary text), produces JSON `{quote, justification}`.
- **Packaging Agent** — assembles markdown, calls one more LLM to rewrite the trend summary in CEO-friendly tone, writes file.

## Multimodal handoff

The copywriter receiving the image is a good study in [[Multimodal Image Input]]:

```python
messages = [
    {"role": "system", "content": "You are a copywriter..."},
    {"role": "user", "content": [
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{b64_img}", "detail": "auto"}},
        {"type": "text", "text": f"""
Trend summary: \"\"\"{trend_summary}\"\"\"
Return JSON: {{"quote": "...", "justification": "..."}}"""},
    ]},
]
```

> [!tip]+ When to use vs. planner-executor
> | Choose static (this pattern)               | Choose dynamic ([[Planner-Executor Architecture]]) |
> | ------------------------------------------ | -------------------------------------------------- |
> | Workflow is known and stable               | Workflow depends on topic / user input             |
> | Each role has a clear, fixed input shape   | Steps and routing emerge from a plan               |
> | You want deterministic order               | You want flexibility per topic                     |
> | Fewer LLM calls, lower cost                | Routing call per step adds cost                    |

> [!warning]+ Trade-offs
> - **Strength:** trivially debuggable. The pipeline function is the whole story.
> - **Limitation:** rigid. Adding a new step means editing the orchestrator function. New routings aren't possible without code changes.
> - **Strength:** mixed frameworks per agent (aisuite for text, raw OpenAI for image) are fine — each agent encapsulates its own SDK choices.

## Seen in

- [[M5 UGL 2]] — the marketing-campaign pipeline (canonical).

## Related

- [[Planner-Executor Architecture]] — dynamic-dispatch alternative.
- [[Multimodal Image Input]] — how the copywriter handles image + text together.
- [[Tool Calling - Manual Loop]] — the `while True` form the market research agent uses.
