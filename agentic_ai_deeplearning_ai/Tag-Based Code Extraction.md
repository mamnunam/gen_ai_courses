---
tags: [technique, agentic-ai, prompt-engineering]
aliases: [execute_python tags, code-block extraction]
---

# Tag-Based Code Extraction

A prompt-output convention where the LLM wraps executable code in a custom tag pair (e.g. `<execute_python>...</execute_python>`) and your code uses a regex to pull it back out. The tag becomes the contract between the LLM and the runtime: anything inside the tag is meant to be executed; everything outside is commentary.

It's a tiny but important pattern — it solves the "how do I separate code from explanation in the model's output" problem with regex instead of trusting the model to return clean code.

## The convention

Pick a unique tag (the labs use `<execute_python>`). Tell the model:

> Return ONLY executable Python between these tags (no extra text):
> ```
> <execute_python>
> # your python
> </execute_python>
> ```

Then extract:

```python
import re

def _extract_execute_block(text: str) -> str:
    """Return Python code inside <execute_python>...</execute_python>."""
    if not text:
        raise RuntimeError("Empty content passed to code executor.")
    m = re.search(r"<execute_python>(.*?)</execute_python>", text,
                  re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()
```

Note the `re.DOTALL` flag — it lets `.` match newlines, which is essential for multi-line code blocks. `re.IGNORECASE` is defensive in case the model alters case.

## Why this and not markdown fences

The standard "```python ... ```" markdown fence works too, but tag-based extraction has advantages:

- **Unique tags don't collide with code content.** Triple-backticks can collide with code that itself contains markdown.
- **Tag-based parsing is unambiguous.** Markdown fences have many variants (\`\`\`, \`\`\`python, \`\`\`py) requiring more lenient regex.
- **You can use multiple tags for multiple purposes.** E.g. `<execute_python>` for runnable code, `<thinking>` for chain-of-thought, etc.

## How the prompt enforces it

You don't just hope the model uses tags — you mandate it as a hard constraint, often near the start and end of the prompt:

```text
Return your answer *strictly* in this format:

<execute_python>
# valid python code here
</execute_python>

Do not add explanations, only the tags and the code.
```

Or, when the output has more than just code (M2_UGL_1's reflection step):

```text
OUTPUT FORMAT (STRICT):
1) First line: a valid JSON object with ONLY the "feedback" field.
2) After a newline, output ONLY the refined Python code wrapped in:
   <execute_python>
   ...
   </execute_python>
```

## Combining with JSON

In `M2_UGL_1`'s `reflect_on_image_and_regenerate`, the model returns a JSON line *and* a tag-wrapped code block. The parser handles them separately:

```python
# Parse the first JSON line (feedback)
lines = content.strip().splitlines()
try:
    obj = json.loads(lines[0])
except Exception:
    m_json = re.search(r"\{.*?\}", content, flags=re.DOTALL)
    obj = json.loads(m_json.group(0)) if m_json else {"feedback": ""}

# Extract code from <execute_python>...</execute_python>
m_code = re.search(r"<execute_python>([\s\S]*?)</execute_python>", content)
refined_code_body = m_code.group(1).strip() if m_code else ""

feedback = obj.get("feedback", "")
```

`[\s\S]*?` is the non-greedy equivalent of `.*?` with multi-line matching baked in — common pattern.

> [!failure]+ What can go wrong
> - **The model adds backticks inside the tag.** Some models still wrap code in \`\`\`python even when told not to. Strip them defensively if needed.
> - **The model omits the tags.** Have a fallback: assume the entire response is code if no tags are found (`_extract_execute_block` does this).
> - **Multiple blocks.** If the model returns two `<execute_python>` blocks, the non-greedy regex grabs the first one only. If you want all, use `re.findall`.

## Seen in

- [[M2 UGL 1]] — chart-code extraction (`generate_chart_code` and `reflect_on_image_and_regenerate`).
- [[M5 UGL 1 R]] — code-as-plan extraction in `_extract_execute_block`.

## Related

- [[Code as Plan]] — the larger pattern this technique supports.
- [[Structured JSON Outputs]] — a parallel technique for non-code structured output.
- [[Sandboxed Execution]] — what you do with the extracted code.
