---
tags: [technique, agentic-ai, prompt-engineering]
aliases: [strict JSON output, JSON contract, prompted JSON]
---

# Structured JSON Outputs

A prompt convention where the LLM is told to respond with **strict JSON only**, often with named fields. Your code then parses it with `json.loads()` and treats it as a typed payload. Used everywhere that one LLM call needs to hand structured data to the next step in a pipeline.

This is the poor-man's alternative to OpenAI's structured-output / function-calling mode — it works on any model and any SDK by being explicit in the prompt and defensive in the parser.

## The minimal contract

The prompt names the keys and gives an example:

```text
Return STRICT JSON with two fields:
{
  "feedback": "<1-3 sentences explaining the gap or confirming correctness>",
  "refined_sql": "<final SQL to run>"
}
```

The parser:

```python
import json

response = client.chat.completions.create(...)
content = response.choices[0].message.content
try:
    obj = json.loads(content)
    feedback     = str(obj.get("feedback", "")).strip()
    refined_sql  = str(obj.get("refined_sql", sql_query)).strip()
    if not refined_sql:
        refined_sql = sql_query
except Exception:
    # Fallback: use raw content as feedback, keep original
    feedback = content.strip()
    refined_sql = sql_query
```

Always have a fallback. JSON parsing fails often enough that you need to handle it.

## Variant: JSON line + tag block

`M2_UGL_1` mixes JSON and tag-wrapped code in one response:

```text
OUTPUT FORMAT (STRICT):
1) First line: a valid JSON object with ONLY the "feedback" field.
   Example: {"feedback": "The legend is unclear and the axis labels overlap."}

2) After a newline, output ONLY the refined Python code wrapped in:
   <execute_python>
   ...
   </execute_python>
```

Then the parser handles them separately — first JSON-parse the first line, then regex-extract the code (see [[Tag-Based Code Extraction]]).

## Variant: triple-fenced JSON

LLMs love to wrap JSON in markdown code fences. Strip them before parsing:

```python
def clean_json_block(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()
```

Used in the `C1M5` executor to clean the router's JSON output.

## Variant: regex-extract first JSON object

When the model insists on adding prose around the JSON, you can pull the first `{...}` block out:

```python
m_json = re.search(r"\{.*?\}", content, flags=re.DOTALL)
if m_json:
    obj = json.loads(m_json.group(0))
```

Less safe (it grabs the first balanced-looking block), but useful for forgiving parses.

## Reflection's signature use

The reflection step in [[Reflection Pattern]] almost always uses structured JSON to separate the critique from the rewrite:

```python
user_prompt = f"""Analyze this report and provide:
1. A structured reflection covering Strengths, Limitations, Suggestions, Opportunities.
2. A revised version of the report.

Output ONLY valid JSON with this exact structure (no extra commentary):
{{
  "reflection": "Your structured reflection here covering the four sections",
  "revised_report": "Your improved version of the report here"
}}"""

response = CLIENT.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are an academic reviewer and editor."},
        {"role": "user",   "content": user_prompt},
    ],
    temperature=temperature,
)
llm_output = response.choices[0].message.content.strip()

try:
    data = json.loads(llm_output)
except json.JSONDecodeError:
    raise Exception("LLM output was not valid JSON. Adjust the prompt.")

return {
    "reflection": str(data.get("reflection", "")).strip(),
    "revised_report": str(data.get("revised_report", "")).strip(),
}
```

## When the model lies about format

Best practices that come up across labs:

- **Repeat the format constraint twice** — once near the top of the prompt, once at the end. Models pay more attention to the end.
- **Use words like "ONLY" and "STRICT".** They actually help: "Output ONLY valid JSON with this exact structure (no additional commentary)."
- **Give an explicit example.** A literal example block of the expected JSON is more reliable than a description.
- **Lower temperature.** `temperature=0` makes format adherence more consistent. For reflection where you want diverse critique content, you might keep `temperature=1.0` and just have a robust fallback.

> [!tip]+ When to use real structured outputs
> If your provider supports it (OpenAI's `response_format={"type": "json_object"}` or Pydantic structured outputs, Anthropic's `tools=`), prefer that — the model is guaranteed to produce valid JSON. The prompt-based approach used here works on any model but is fallback-heavier.

## Seen in

- [[M2 UGL 1]] — JSON feedback + tag-wrapped code in one response.
- [[M2 UGL 2]] — JSON `{"feedback", "refined_sql"}` for SQL reflection.
- [[C1M3 Assignment]] — JSON `{"reflection", "revised_report"}`.
- [[C1M5 Assignment]] — JSON `{"agent", "task"}` for the per-step router; `clean_json_block` helper.
- [[M5 UGL 2]] — JSON `{"prompt", "caption"}` from designer, `{"quote", "justification"}` from copywriter.
