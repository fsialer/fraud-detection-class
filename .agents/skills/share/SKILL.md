---
name: share
description: Export a conversation summary as a learning log for instructor review
user_invocable: true
---

Write a conversation summary to `conversations/` in the project root. The file should be named with today's date and a short slug, e.g. `conversations/2026-03-21-etl-pipeline.md`.

Create the `conversations/` directory if it doesn't exist.

The summary should include:

1. **Issue worked on** — which GitHub issue number and title
2. **Approach discussed** — what approaches were considered and why the chosen one was selected
3. **Key decisions** — any design decisions, trade-offs, or things the user learned
4. **What was built** — brief description of files created/modified
5. **Open questions** — anything unresolved or to revisit

Keep it concise — this is a learning log, not a transcript. Focus on the reasoning and decisions, not the code itself.

After writing the file, stage it with `git add` and let the user know the path so they can commit it.
