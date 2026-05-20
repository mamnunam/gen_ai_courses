---
tags: [moc, index, agentic-ai]
aliases: [home, map of content, course index]
---

# DeepLearning.AI — Agentic AI Course (Notes Vault)

Concept-organized notes from the **DeepLearning.AI Agentic AI** course (10 notebooks across Modules 2-5). Each linked note is an atomic concept — pattern, technique, or piece of infrastructure — distilled from the labs.

The course is intentionally **framework-agnostic** (no LangChain / LangGraph / CrewAI / AutoGen). Everything is built from `aisuite` or the raw OpenAI SDK, so the patterns transfer.

---

## 🧭 Core Patterns

The big agentic design patterns the course teaches. Start here.

- [[Reflection Pattern]] — generate → critique → revise. The foundation.
- [[External Feedback Reflection]] — grounding the critic in execution output, not just text.
- [[Tool Calling - Auto Orchestration]] — let `aisuite` handle the tool-call loop.
- [[Tool Calling - Manual Loop]] — handle the `tool_calls` protocol yourself.
- [[Code as Plan]] — the LLM writes Python that *is* the plan.
- [[Planner-Executor Architecture]] — plan → per-step router → specialist agents.
- [[Sequential Multi-Agent Pipeline]] — static DAG of specialized agents.

## 🛠️ Techniques

Reusable tactics that show up across patterns.

- [[Tag-Based Code Extraction]] — `<execute_python>` regex.
- [[Sandboxed Execution]] — `exec()` with a controlled namespace.
- [[Structured JSON Outputs]] — strict JSON contracts in prompts.
- [[Schema Injection Prompting]] — putting your data shape in the prompt.
- [[Multimodal Image Input]] — passing base64 images into the LLM.
- [[Mixed-Model Strategy]] — cheap-gen, strong-critic.
- [[Agent Registry and Routing]] — dict-of-callables dispatch.

## 🎯 Concepts

Ideas worth their own atomic note even though they aren't recipes.

- [[Component-Level Evaluation]] — evaluating one piece of a pipeline.
- [[Capability Boundary]] — tools define what an agent can do.
- [[Context Accumulation]] — passing prior step outputs forward.

## ⚙️ Infrastructure

The SDKs and modules underneath the patterns.

- [[aisuite Client]] — Andrew Ng's unified-provider client.
- [[OpenAI Tool Call Protocol]] — the wire format aisuite hides.
- [[Research Tools Module]] — arxiv / tavily / wikipedia wrappers.

---

## 📓 Source Notebooks

| Notebook | Module | Type | Main focus |
| --- | --- | --- | --- |
| [[M2 UGL 1]] | 2 | Ungraded | Chart generation with reflection (image-grounded) |
| [[M2 UGL 2]] | 2 | Ungraded | SQL generation with execution-grounded reflection |
| [[C1M2 Assignment]] | 2 | Graded | Three-step reflection on essays |
| [[M3 UGL 1]] | 3 | Ungraded | Functions → tools (auto and manual) |
| [[M3 UGL 2]] | 3 | Ungraded | Email assistant + capability boundary |
| [[C1M3 Assignment]] | 3 | Graded | Manual tool loop + reflection + HTML |
| [[M4 UGL 1]] | 4 | Ungraded | Component-level eval on web search |
| [[M5 UGL 1 R]] | 5 | Ungraded | Code-as-plan customer service agent |
| [[M5 UGL 2]] | 5 | Ungraded | Multi-agent marketing pipeline |
| [[C1M5 Assignment]] | 5 | Graded | Planner-executor multi-agent system |

---

## 🗺️ Suggested Reading Paths

### "Just the patterns"
[[Reflection Pattern]] → [[Tool Calling - Auto Orchestration]] → [[Code as Plan]] → [[Planner-Executor Architecture]] → [[Sequential Multi-Agent Pipeline]]

### "Building up from primitives"
[[aisuite Client]] → [[OpenAI Tool Call Protocol]] → [[Tool Calling - Manual Loop]] → [[Tool Calling - Auto Orchestration]] → [[Capability Boundary]]

### "Reflection deep dive"
[[Reflection Pattern]] → [[Structured JSON Outputs]] → [[External Feedback Reflection]] → [[Multimodal Image Input]] → [[Mixed-Model Strategy]]

### "Code-as-plan deep dive"
[[Schema Injection Prompting]] → [[Tag-Based Code Extraction]] → [[Sandboxed Execution]] → [[Code as Plan]]

### "Multi-agent systems"
[[Sequential Multi-Agent Pipeline]] → [[Agent Registry and Routing]] → [[Context Accumulation]] → [[Planner-Executor Architecture]]

### "Evaluation"
[[Component-Level Evaluation]] → [[External Feedback Reflection]]

---

## 🔗 Concept × Notebook Matrix

A quick way to find which notebook teaches a given concept.

| Concept | Notebooks |
| --- | --- |
| [[Reflection Pattern]] | [[M2 UGL 1]], [[M2 UGL 2]], [[C1M2 Assignment]], [[C1M3 Assignment]] |
| [[External Feedback Reflection]] | [[M2 UGL 2]] (canonical), [[M2 UGL 1]], [[M5 UGL 1 R]], [[M4 UGL 1]] |
| [[Tool Calling - Auto Orchestration]] | [[M3 UGL 1]], [[M3 UGL 2]], [[M4 UGL 1]], [[C1M5 Assignment]], [[M5 UGL 2]] |
| [[Tool Calling - Manual Loop]] | [[C1M3 Assignment]] (canonical), [[M3 UGL 1]], [[M5 UGL 2]] |
| [[Code as Plan]] | [[M5 UGL 1 R]] |
| [[Planner-Executor Architecture]] | [[C1M5 Assignment]] |
| [[Sequential Multi-Agent Pipeline]] | [[M5 UGL 2]] |
| [[Tag-Based Code Extraction]] | [[M2 UGL 1]], [[M5 UGL 1 R]] |
| [[Sandboxed Execution]] | [[M2 UGL 1]], [[M5 UGL 1 R]] |
| [[Structured JSON Outputs]] | All notebooks except [[M3 UGL 1]] and [[C1M2 Assignment]] |
| [[Schema Injection Prompting]] | [[M2 UGL 1]], [[M2 UGL 2]], [[M5 UGL 1 R]] |
| [[Multimodal Image Input]] | [[M2 UGL 1]], [[M5 UGL 2]] |
| [[Mixed-Model Strategy]] | [[M2 UGL 1]], [[M2 UGL 2]], [[C1M2 Assignment]], [[M5 UGL 2]] |
| [[Agent Registry and Routing]] | [[C1M5 Assignment]], [[C1M3 Assignment]] (tools variant) |
| [[Component-Level Evaluation]] | [[M4 UGL 1]] |
| [[Capability Boundary]] | [[M3 UGL 2]] |
| [[Context Accumulation]] | [[C1M5 Assignment]], [[C1M3 Assignment]], [[M5 UGL 2]] |
| [[aisuite Client]] | Most notebooks |
| [[OpenAI Tool Call Protocol]] | [[C1M3 Assignment]], [[M3 UGL 1]] |
| [[Research Tools Module]] | [[M4 UGL 1]], [[C1M3 Assignment]], [[C1M5 Assignment]] |

---

## 🧱 Notebook → Concept Index

If you'd rather start from a notebook:

**Module 2 — Reflection**
- [[M2 UGL 1]]: [[Reflection Pattern]], [[Multimodal Image Input]], [[Tag-Based Code Extraction]], [[Sandboxed Execution]], [[Structured JSON Outputs]], [[Schema Injection Prompting]], [[Mixed-Model Strategy]]
- [[M2 UGL 2]]: [[Reflection Pattern]], [[External Feedback Reflection]], [[Schema Injection Prompting]], [[Structured JSON Outputs]]
- [[C1M2 Assignment]]: [[Reflection Pattern]], [[Mixed-Model Strategy]], [[aisuite Client]]

**Module 3 — Tools**
- [[M3 UGL 1]]: [[Tool Calling - Auto Orchestration]], [[Tool Calling - Manual Loop]], [[OpenAI Tool Call Protocol]], [[aisuite Client]]
- [[M3 UGL 2]]: [[Tool Calling - Auto Orchestration]], [[Capability Boundary]]
- [[C1M3 Assignment]]: [[Tool Calling - Manual Loop]], [[OpenAI Tool Call Protocol]], [[Research Tools Module]], [[Reflection Pattern]], [[Structured JSON Outputs]], [[Agent Registry and Routing]] (TOOL_MAPPING form)

**Module 4 — Evaluation**
- [[M4 UGL 1]]: [[Component-Level Evaluation]], [[Tool Calling - Auto Orchestration]], [[Research Tools Module]]

**Module 5 — Planning & Multi-Agent**
- [[M5 UGL 1 R]]: [[Code as Plan]], [[Sandboxed Execution]], [[Tag-Based Code Extraction]], [[Schema Injection Prompting]], [[External Feedback Reflection]]
- [[M5 UGL 2]]: [[Sequential Multi-Agent Pipeline]], [[Multimodal Image Input]], [[Tool Calling - Manual Loop]], [[Structured JSON Outputs]]
- [[C1M5 Assignment]]: [[Planner-Executor Architecture]], [[Agent Registry and Routing]], [[Context Accumulation]], [[Tool Calling - Auto Orchestration]], [[Research Tools Module]]

---

## 📦 What's Not in the Course

The course is deliberately framework-free. If you want to compare, here's what's covered elsewhere but **not** taught here:

- LangChain / LangGraph / CrewAI / AutoGen frameworks
- OpenAI's `response_format` / Pydantic structured outputs (the course uses prompt-based JSON contracts instead)
- Production retrievers / RAG over vector DBs
- Full sandboxing (real isolation, not just `exec` with a namespace)
- Async agent execution, parallel tool calls, agent-to-agent communication protocols

Those are reasonable next stops once these primitives are familiar.

---

*Vault notes derived from the DeepLearning.AI Agentic AI course notebooks (M2-M5). Concept-organized for atomic Obsidian notes.*
