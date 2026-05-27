---
allowed-tools: Bash(uv run:lab *), Bash(uv run lab *), Bash(ls *), Bash(head *), Read
description: Show a finding's frontmatter + TL;DR, sync its DB row
argument-hint: <id>  (e.g. 005, F-005, 005-12gb-agent)
---

## Your task

The user wants to see finding `$ARGUMENTS`.

1. Normalize the id. Accept any of:
   - `005`               → `F-005`
   - `F-005`             → `F-005`
   - `005-12gb-agent`    → `F-005`
   Pad numbers to 3 digits when they're shorter.

2. Resolve the file with a glob: `docs/findings/F-NNN-*.md`. If multiple
   match (rare — should be one), show the user the list and ask. If zero,
   list `ls docs/findings/F-*.md` and stop.

3. Re-sync the findings table so the DB matches the on-disk doc:

```bash
uv run lab findings sync
```

4. Read the matched markdown file. Print:
   - The frontmatter block (everything between `---` lines, if present)
   - The H1 line
   - The first H2 section (usually `## TL;DR` or `## Claim`) verbatim

Keep the output focused; do not dump the full file. If the file is
small (< 60 lines), it's fine to show the entire thing.
