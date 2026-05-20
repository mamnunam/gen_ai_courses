---
tags: [technique, agentic-ai, multimodal]
aliases: [vision LLM input, image_url, base64 image]
---

# Multimodal Image Input

Passing images into an LLM call alongside text. The standard wire format is a multi-part `content` array on a `user` message, where one part is a `text` block and another is an `image_url` block carrying either a URL or a base64-encoded data URI.

Used in the course in two places: passing a rendered chart to a vision model for critique (reflection), and passing a generated campaign image to a copywriter agent.

## The OpenAI/aisuite shape

The `content` field becomes a list of typed parts instead of a plain string:

```python
import base64

with open(image_path, "rb") as f:
    img_bytes = f.read()
b64_img = base64.b64encode(img_bytes).decode("utf-8")

messages = [
    {
        "role": "system",
        "content": "You are a copywriter that creates elegant campaign quotes."
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64_img}",
                    "detail": "auto"
                }
            },
            {
                "type": "text",
                "text": f"""
Here is a visual marketing image and a trend analysis:

Trend summary: \"\"\"{trend_summary}\"\"\"

Please return a JSON object like:
{{"quote": "...", "justification": "..."}}"""
            }
        ]
    }
]

response = client.chat.completions.create(model=model, messages=messages)
```

Two things to note:

- The `image_url.url` is a **data URI** with the base64 payload inline. Equally valid to pass an `https://` URL if the image is hosted.
- `detail: "auto"` lets the model pick low/high resolution. Use `"high"` for charts where you want the model to read axis labels.

## Helper for vision-call abstraction

`M2_UGL_1` factors the call behind a `utils.encode_image_b64` + provider-specific helpers:

```python
media_type, b64 = utils.encode_image_b64(chart_path)

lower = model_name.lower()
if "claude" in lower or "anthropic" in lower:
    content = utils.image_anthropic_call(model_name, prompt, media_type, b64)
else:
    content = utils.image_openai_call(model_name, prompt, media_type, b64)
```

Why the split: OpenAI uses `image_url` with a data URI, Anthropic uses `image` with a separate `source` object. Different providers, same idea. A helper module hides the difference.

## Use case 1 — image as reflection input

In `M2_UGL_1`, the workflow is: generate matplotlib code → execute it → save `chart_v1.png` → pass the image to a vision LLM for critique.

The critic literally *sees* the chart and can say things like:
- "The legend is unclear and the axis labels overlap."
- "The y-axis is missing a unit label."
- "The two series are nearly the same color; choose distinct hues."

These critiques are impossible from the code alone — they require seeing the rendered output. See [[External Feedback Reflection]] for the broader pattern.

## Use case 2 — image as semantic context

In `M5_UGL_2`, the copywriter agent is given the generated campaign image *along with* the text trend summary, and asked for a quote that fits the image. Now the LLM isn't critiquing the image — it's *interpreting* it to inform a text generation.

## Producing an image (not just consuming one)

A bridging pattern: aisuite (at lab time) doesn't support image generation, so `M5_UGL_2`'s graphic designer agent uses the raw OpenAI SDK for that step:

```python
import openai

openai_client = openai.OpenAI()
image_response = openai_client.images.generate(
    model="gpt-image-1-mini",
    prompt=prompt,
    size="1024x1024",
    quality="medium",
    n=1,
    # no response_format — gpt-image-1 always returns b64_json
)

b64 = image_response.data[0].b64_json
img_bytes = base64.b64decode(b64)
img = Image.open(BytesIO(img_bytes))
img.save("generated_image.png")
```

The point: a single workflow can use aisuite for text and raw OpenAI for image generation. Wrap each in its own agent function.

> [!failure]+ What can go wrong
> - **Too small / too low-res** — the model can't read text on the image. Increase resolution or use `detail: "high"`.
> - **Confused about which image is which** — when passing multiple images, label them in the text block ("Image A above is the V1 chart; Image B is the reference design.").
> - **Hallucinated content** — the model will sometimes describe things that aren't in the image. The reflection prompt should ask it to ground its critique in specific visual elements.

## Seen in

- [[M2 UGL 1]] — `reflect_on_image_and_regenerate` passes the V1 chart image to a critic.
- [[M5 UGL 2]] — `copywriter_agent` takes the generated campaign image and writes a quote for it.

## Related

- [[External Feedback Reflection]] — image-as-feedback is the broader pattern.
- [[Sequential Multi-Agent Pipeline]] — the marketing pipeline's image hand-off.
