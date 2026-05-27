---
allowed-tools: Bash(git add:*), Bash(git commit:*), Bash(git status:*), Read, Edit
description: Promote a model card from "experimental" to "active" status (frontmatter + commit)
argument-hint: <litellm_id>  (e.g. qwen3-14b-q4)
---

## Your task

The user wants to promote model `$ARGUMENTS` from `status: experimental`
to `status: active` in its model card.

1. Read `docs/model-cards/$ARGUMENTS.md`. If the file does not exist:
   - Show `ls docs/model-cards/ | head -30` and ask which model they
     meant. Stop.

2. Inspect the frontmatter `status:` line. It should currently be
   `status: experimental`.
   - If it's already `status: active`: tell the user, no change needed, stop.
   - If it's something else (retired, deprecated, etc.): ask the user to
     confirm the transition before editing.

3. Use the Edit tool to replace exactly:
   ```
   status: experimental
   ```
   with:
   ```
   status: active
   ```
   inside `docs/model-cards/$ARGUMENTS.md`.

4. Stage and commit:

```bash
git add docs/model-cards/$ARGUMENTS.md
git commit -m "model-cards: promote $ARGUMENTS from experimental to active"
```

5. Print the resulting commit SHA. Do NOT push.
