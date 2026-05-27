---
allowed-tools: Bash(uv run:python *), Bash(uv run python *)
description: Scaffold a new lab experiment (EXP-NNN doc + sweep config + analysis dir + DB row)
argument-hint: <slug>
---

## Your task

The user wants to scaffold a new experiment with the slug `$ARGUMENTS`.

Run the lab experiment-scaffolding tool from the repo root:

```bash
uv run python tools/new_experiment.py $ARGUMENTS
```

This will:
- Assign the next available EXP-NNN number
- Create `docs/exp/EXP-NNN-<slug>.md` from `docs/exp/_template.md`
- Create `conf/sweep/EXP-NNN.yaml` from `conf/sweep/_template.yaml`
- Create `analysis/EXP-NNN/.gitkeep`
- Insert a placeholder row in `experiments` (status=`planned`, no `plan_git_sha`)

After it runs, show the user the printed "Next steps" and stop. Do NOT
edit the generated doc/sweep yourself unless the user explicitly asks.

If the user did not provide a slug, ask them for one (lowercase
letters / digits / hyphens, 2-64 chars).
