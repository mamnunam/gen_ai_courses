---
tags: [concept, agentic-ai, tools]
aliases: [tool boundary, agent capability]
---

# Capability Boundary

The list of tools you expose to an agent **is** the set of things it can do in the world. No matter how clever the LLM is, if `delete_email` isn't in its tools list, it cannot delete emails — it will reason about the task, narrate what it would do, and stop.

This is obvious in retrospect but worth naming because it changes how you design agents: capability design is tool-list design.

## The lab's demonstration (M3_UGL_2)

Same prompt, two different tool lists:

**Without `delete_email`** — the agent reasons but cannot act:

```python
prompt_ = build_prompt("Delete alice@work.com email")

response = client.chat.completions.create(
    model="openai:o4-mini",
    messages=[{"role": "user", "content": prompt_}],
    tools=[                              # ← delete_email is absent
        email_tools.search_unread_from_sender,
        email_tools.list_unread_emails,
        email_tools.search_emails,
        email_tools.get_email,
        email_tools.mark_email_as_read,
        email_tools.send_email,
    ],
    max_turns=5,
)
```

The agent searches for the email, finds it, and then ... reports back that it doesn't have a way to delete. It cannot do the task.

**Add `email_tools.delete_email`** to the list, run again. Now the agent searches → finds → deletes. Same prompt, same model — different capabilities.

> [!info]+ Why this matters for design
> When you add an agent to a system, the design question isn't "what should the agent be smart about?" — it's **"what tools should it have access to?"** Because:
>
> - The agent cannot exceed its tools. So safety scopes down naturally if you scope the tool list.
> - The agent's reliability is bounded by tool reliability. A flaky `send_email` tool means a flaky agent.
> - The agent's transparency is shaped by tool granularity. Many small tools = clear traces; one God-tool = opaque behavior.

> [!tip]+ Practical implications
> - **Principle of least authority.** Give the agent only the tools it needs for the current task. Don't add `delete_email` to a research agent.
> - **Read vs. write tools.** Splitting `list_emails` (read) from `delete_email` (write) lets you grant read-only access easily.
> - **Capability shifts during a session.** You can vary the tool list per request based on user permissions ("you can only delete your own emails").
> - **Failure mode signal.** When the agent reasons-but-doesn't-act, check the tool list before debugging the prompt.

## The agent reasons regardless

Even without the tool, the LLM does *something*. In the email lab, the agent narrates "I'll look at the available tools... I can find the email but cannot delete it. Please use a different tool." That narration is useful — it tells the user what's missing — but it's not the same as completing the task.

## Connection to safety

[[Code as Plan]] flips this trade-off. Instead of curating a tool list, you give the model Python + a few helpers. Capability grows enormously; safety has to come from [[Sandboxed Execution]] and prompt-level policy ("don't mutate inventory unless intent is clearly purchase/return"). The boundary moves from "what's in the tool list" to "what's in the controlled namespace + what the prompt allows."

For [[Tool Calling - Auto Orchestration]], capability is the tool list. For code-as-plan, capability is the namespace + the prompt policy.

## Seen in

- [[M3 UGL 2]] — the canonical demonstration: same prompt, with and without `delete_email`.

## Related

- [[Tool Calling - Auto Orchestration]] — the mechanism that creates the boundary.
- [[Code as Plan]] — the alternative model where capability isn't a curated list.
