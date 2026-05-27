---
allowed-tools: Bash(uv run:lab *), Bash(uv run lab *), Bash(ls *), Bash(cat *)
description: Run a lab sweep with pre-registration enforcement
argument-hint: <slug>  (e.g. EXP-001, 002, 003b, reliability-001)
---

## Your task

The user wants to run the sweep for `$ARGUMENTS`.

1. Resolve the sweep config path:
   - If `$ARGUMENTS` already matches `conf/sweep/<name>.yaml`, use it as-is.
   - Otherwise try `conf/sweep/EXP-$ARGUMENTS.yaml`,
     `conf/sweep/EXP-$ARGUMENTS-pilot.yaml`, then `conf/sweep/$ARGUMENTS.yaml`.
   - If none exist, show the user `ls conf/sweep/` and ask which they meant.

2. Confirm the resolved path with the user before running (`ls -l <path>`).

3. Invoke the sweep with pre-registration enforced:

```bash
uv run lab sweep run conf/sweep/EXP-<slug>.yaml --enforce-pre-registration
```

If the sweep refuses with "experiment ... is not pre-registered", remind the
user they need to register the plan first:

```bash
uv run lab exp register docs/exp/EXP-<slug>.md
```

4. After the sweep starts (or completes synchronously), report the summary
   line `lab sweep run` prints. Do NOT analyze results — that's a separate
   step (run `scripts/analyze_exp<NNN>.py` or `uv run lab analyze`).

If the user did not provide a slug, list `conf/sweep/EXP-*.yaml` and ask.
