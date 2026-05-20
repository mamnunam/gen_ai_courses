---
tags: [infrastructure, agentic-ai, tools, search]
aliases: [research_tools, arxiv tavily wikipedia]
---

# Research Tools Module

A lab-local Python module (`research_tools.py`) that wraps three search backends — **arXiv**, **Tavily** (general web), and **Wikipedia** — into Python functions that can be passed straight to an LLM as tools. The course's stand-in for "let the agent search the world."

Used as the toolset for the research agents in M4, C1M3, and C1M5.

## The three tools

| Function                                            | Backend       | Returns                                                  |
| --------------------------------------------------- | ------------- | -------------------------------------------------------- |
| `arxiv_search_tool(query, max_results=5)`           | arXiv API     | list of papers: `title, authors, published, summary, url, link_pdf` |
| `tavily_search_tool(query, max_results, include_images)` | Tavily API    | list of web results: `title, content, url` (+ optional image URLs) |
| `wikipedia_search_tool(query)`                      | Wikipedia API | encyclopedic summaries                                   |

## Calling them directly

```python
import research_tools

# arXiv
arxiv_results = research_tools.arxiv_search_tool("linear algebra", max_results=3)
for paper in arxiv_results:
    print(paper['title'], paper['url'])

# Tavily
tavily_results = research_tools.tavily_search_tool("retrieval-augmented generation applications")
for item in tavily_results:
    print(item['title'], item['url'])

# Wikipedia
wiki_results = research_tools.wikipedia_search_tool("Ensemble Kalman Filter")
```

## Using as LLM tools (aisuite auto)

The simplest form — pass the functions to `aisuite`:

```python
response = client.chat.completions.create(
    model="openai:gpt-4o",
    messages=[{"role": "user", "content": f"""
        You are a research function with access to:
        - arxiv_tool: academic papers
        - tavily_tool: general web search
        - wikipedia_tool: encyclopedic summaries

        Task: Find 2 recent papers about black hole science.
        Today is {datetime.now().strftime('%Y-%m-%d')}.
    """}],
    tools=[
        research_tools.arxiv_search_tool,
        research_tools.tavily_search_tool,
        research_tools.wikipedia_search_tool,
    ],
    tool_choice="auto",
    max_turns=5,
)
```

The model picks among the three based on the task: arXiv for academic, Tavily for general/recent web, Wikipedia for definitions and background.

## Using as LLM tools (raw OpenAI manual)

`C1M3` uses the raw SDK with explicit tool schemas and a `TOOL_MAPPING`:

```python
TOOL_MAPPING = {
    "tavily_search_tool": research_tools.tavily_search_tool,
    "arxiv_search_tool":  research_tools.arxiv_search_tool,
}

tools = [research_tools.arxiv_tool_def, research_tools.tavily_tool_def]
# arxiv_tool_def / tavily_tool_def are pre-built JSON schemas
```

Wikipedia is omitted here — `C1M3` uses just two tools.

## Telling the model what each tool is for

Prompt convention: list the tools by what they're best at, not by their interface:

```text
You can use the following tools:
- arxiv_tool to find academic papers
- tavily_tool for general web searches
- wikipedia_tool for accessing encyclopedic knowledge
```

This is more effective than a bland "you have three search tools" — it guides selection.

## Date awareness

A consistent gotcha: search results are time-sensitive but the LLM doesn't know what today is. The labs always inject the date:

```python
prompt = f"""
You can use the following tools:
- arxiv_tool, tavily_tool, wikipedia_tool

Task: {task}.
Today is {datetime.now().strftime('%Y-%m-%d')}.
"""
```

For "recent papers," "current trends," etc., this is the difference between a useful answer and an outdated one.

## Where it powers the labs

- [[M4 UGL 1]] — wrapped in `find_references()` and evaluated for preferred domains.
- [[C1M3 Assignment]] — paired with [[Tool Calling - Manual Loop]] to generate a research report.
- [[C1M5 Assignment]] — the research_agent's toolset.

## Tavily-specific notes

Tavily is the "general web" backend. It's an alternative to direct Google search and tends to return citation-friendly results (title, URL, content snippet). For [[Component-Level Evaluation]] in M4, Tavily's URLs are checked against a `TOP_DOMAINS` allowlist — a useful eval signal because web search quality is the most variable in this stack.

## Related

- [[Tool Calling - Auto Orchestration]] — the main consumption path.
- [[Tool Calling - Manual Loop]] — the C1M3 path.
- [[Component-Level Evaluation]] — eval on the URLs Tavily returns.
