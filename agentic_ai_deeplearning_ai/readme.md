# DeepLearning.AI — Agentic AI Course Notes
 
Concept-organized notes from the **DeepLearning.AI Agentic AI** course (Modules 2-5, 10 notebooks). Written as an Obsidian vault — atomic notes, one per concept, heavily cross-linked. Renders fine on GitHub too, with a couple of caveats noted below.
 
The course is intentionally framework-free (no LangChain, LangGraph, CrewAI, or AutoGen). Everything is built from `aisuite` or the raw OpenAI SDK, so the patterns transfer directly to any stack.
 
## What's inside
 
```
.
├── _Index.md              ← Map of Content (start here in Obsidian)
├── README.md              ← This file (start here on GitHub)
├── <20 concept notes>.md  ← One atomic note per concept
└── Notebooks/
    └── <10 notebook stubs>.md  ← Thin reference notes mapping notebook → concepts
```
 
20 atomic concept notes split across four buckets — patterns, techniques, concepts, infrastructure — plus 10 thin notebook stubs that act as breadcrumbs back to the source material.
 
## How to use it
 
### As an Obsidian vault (recommended)
 
Point Obsidian at the repo folder and open `_Index.md`. The vault uses:
 
- `[[Wiki-links]]` between notes — Obsidian resolves these to the matching `.md` files.
- YAML frontmatter with `tags` and `aliases` — surfaces in the properties panel and link auto-suggest.
- Obsidian callouts (`> [!tip]`, `> [!info]`, `> [!warning]`, `> [!failure]`) — render as colored boxes in reading view, summarising when-to-use / why-it-works / trade-offs / pitfalls.
Everything is plain markdown — no plugins required.
 
### As a GitHub repository (reading on the web)
 
Browse the file tree or use the concept index below. Two rendering caveats on GitHub:
 
- **Wiki-links** (`[[Reflection Pattern]]`) render as literal text, not clickable links. Use the **concept index below** for navigation between notes — the standard markdown links work everywhere.
- **Callouts**: GitHub only supports `[!NOTE]`, `[!TIP]`, `[!IMPORTANT]`, `[!WARNING]`, `[!CAUTION]`. The vault uses `[!tip]` and `[!warning]` (which GitHub renders correctly) plus `[!info]` and `[!failure]` (which GitHub falls back to plain blockquotes). Content is still readable either way.
## Concept index
 
### Core Patterns
 
The big agentic design patterns. Start here.
 
- [Reflection Pattern](Reflection%20Pattern.md) — generate → critique → revise. The foundation.
- [External Feedback Reflection](External%20Feedback%20Reflection.md) — grounding the critic in execution output, not just text.
- [Tool Calling - Auto Orchestration](Tool%20Calling%20-%20Auto%20Orchestration.md) — let `aisuite` handle the tool-call loop.
- [Tool Calling - Manual Loop](Tool%20Calling%20-%20Manual%20Loop.md) — handle the `tool_calls` protocol yourself.
- [Code as Plan](Code%20as%20Plan.md) — the LLM writes Python that *is* the plan.
- [Planner-Executor Architecture](Planner-Executor%20Architecture.md) — plan → per-step router → specialist agents.
- [Sequential Multi-Agent Pipeline](Sequential%20Multi-Agent%20Pipeline.md) — static DAG of specialized agents.
### Techniques
 
Reusable tactics that show up across patterns.
 
- [Tag-Based Code Extraction](Tag-Based%20Code%20Extraction.md) — `<execute_python>` regex.
- [Sandboxed Execution](Sandboxed%20Execution.md) — `exec()` with a controlled namespace.
- [Structured JSON Outputs](Structured%20JSON%20Outputs.md) — strict JSON contracts in prompts.
- [Schema Injection Prompting](Schema%20Injection%20Prompting.md) — putting your data shape in the prompt.
- [Multimodal Image Input](Multimodal%20Image%20Input.md) — passing base64 images into the LLM.
- [Mixed-Model Strategy](Mixed-Model%20Strategy.md) — cheap-gen, strong-critic.
- [Agent Registry and Routing](Agent%20Registry%20and%20Routing.md) — dict-of-callables dispatch.
### Concepts
 
Ideas worth their own atomic note even though they aren't recipes.
 
- [Component-Level Evaluation](Component-Level%20Evaluation.md) — evaluating one piece of a pipeline.
- [Capability Boundary](Capability%20Boundary.md) — tools define what an agent can do.
- [Context Accumulation](Context%20Accumulation.md) — passing prior step outputs forward.
### Infrastructure
 
The SDKs and modules underneath the patterns.
 
- [aisuite Client](aisuite%20Client.md) — Andrew Ng's unified-provider client.
- [OpenAI Tool Call Protocol](OpenAI%20Tool%20Call%20Protocol.md) — the wire format aisuite hides.
- [Research Tools Module](Research%20Tools%20Module.md) — arxiv / tavily / wikipedia wrappers.
## Source notebooks
 
| Notebook | Module | Type | Main focus |
| --- | --- | --- | --- |
| [M2 UGL 1](Notebooks/M2%20UGL%201.md) | 2 | Ungraded | Chart generation with reflection (image-grounded) |
| [M2 UGL 2](Notebooks/M2%20UGL%202.md) | 2 | Ungraded | SQL generation with execution-grounded reflection |
| [C1M2 Assignment](Notebooks/C1M2%20Assignment.md) | 2 | Graded | Three-step reflection on essays |
| [M3 UGL 1](Notebooks/M3%20UGL%201.md) | 3 | Ungraded | Functions → tools (auto and manual) |
| [M3 UGL 2](Notebooks/M3%20UGL%202.md) | 3 | Ungraded | Email assistant + capability boundary |
| [C1M3 Assignment](Notebooks/C1M3%20Assignment.md) | 3 | Graded | Manual tool loop + reflection + HTML |
| [M4 UGL 1](Notebooks/M4%20UGL%201.md) | 4 | Ungraded | Component-level eval on web search |
| [M5 UGL 1 R](Notebooks/M5%20UGL%201%20R.md) | 5 | Ungraded | Code-as-plan customer service agent |
| [M5 UGL 2](Notebooks/M5%20UGL%202.md) | 5 | Ungraded | Multi-agent marketing pipeline |
| [C1M5 Assignment](Notebooks/C1M5%20Assignment.md) | 5 | Graded | Planner-executor multi-agent system |
 
## Suggested reading paths
 
**Just the patterns**
[Reflection Pattern](Reflection%20Pattern.md) → [Tool Calling - Auto Orchestration](Tool%20Calling%20-%20Auto%20Orchestration.md) → [Code as Plan](Code%20as%20Plan.md) → [Planner-Executor Architecture](Planner-Executor%20Architecture.md) → [Sequential Multi-Agent Pipeline](Sequential%20Multi-Agent%20Pipeline.md)
 
**Building up from primitives**
[aisuite Client](aisuite%20Client.md) → [OpenAI Tool Call Protocol](OpenAI%20Tool%20Call%20Protocol.md) → [Tool Calling - Manual Loop](Tool%20Calling%20-%20Manual%20Loop.md) → [Tool Calling - Auto Orchestration](Tool%20Calling%20-%20Auto%20Orchestration.md) → [Capability Boundary](Capability%20Boundary.md)
 
**Reflection deep dive**
[Reflection Pattern](Reflection%20Pattern.md) → [Structured JSON Outputs](Structured%20JSON%20Outputs.md) → [External Feedback Reflection](External%20Feedback%20Reflection.md) → [Multimodal Image Input](Multimodal%20Image%20Input.md) → [Mixed-Model Strategy](Mixed-Model%20Strategy.md)
 
**Code-as-plan deep dive**
[Schema Injection Prompting](Schema%20Injection%20Prompting.md) → [Tag-Based Code Extraction](Tag-Based%20Code%20Extraction.md) → [Sandboxed Execution](Sandboxed%20Execution.md) → [Code as Plan](Code%20as%20Plan.md)
 
**Multi-agent systems**
[Sequential Multi-Agent Pipeline](Sequential%20Multi-Agent%20Pipeline.md) → [Agent Registry and Routing](Agent%20Registry%20and%20Routing.md) → [Context Accumulation](Context%20Accumulation.md) → [Planner-Executor Architecture](Planner-Executor%20Architecture.md)
 
**Evaluation**
[Component-Level Evaluation](Component-Level%20Evaluation.md) → [External Feedback Reflection](External%20Feedback%20Reflection.md)
 
## What's *not* covered
 
The course is deliberately framework-free. If you're comparing to other agentic resources, the following are intentionally absent:
 
- LangChain / LangGraph / CrewAI / AutoGen frameworks
- OpenAI's structured outputs / Pydantic structured outputs (the course uses prompt-based JSON contracts instead)
- Production retrievers / RAG over vector DBs
- Real sandboxing (the course uses `exec` with a controlled namespace, not OS-level isolation)
- Async agent execution, parallel tool calls, agent-to-agent communication protocols
Reasonable next stops once these primitives are familiar.
 
## About these notes
 
These are personal study notes, not official course material. They're distilled from the lab notebooks but reorganized atomically — one note per concept rather than one note per lab — so the same idea isn't restated three times across modules.
 
The course itself ([DeepLearning.AI Agentic AI](https://www.deeplearning.ai/)) is the source of truth for the lab content. These notes exist for reference and review, not as a substitute.
 
## License
 
Notes are MIT-licensed for personal study use. The course content and lab notebooks themselves remain the property of DeepLearning.AI.
