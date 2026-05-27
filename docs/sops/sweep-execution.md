---
doc_id: sweep-execution
title: 'SOP: Sweep execution'
zone: lab
kind: guide
status: active
owner: m
created: '2026-05-25'
last_updated: '2026-05-25'
last_verified: '2026-05-25'
tags:
- lab
- guide
- sops
---
# SOP: Sweep execution

**Purpose:** end-to-end procedure for executing a comparison sweep, from plan to analysis.

**Pre-conditions:** lab services up (`MinIO`, `MLflow`, `LiteLLM`, `Postgres`, `Valkey`, `Ollama`). At least one task suite registered. Target models registered in `lab.models`.

---

## 1. Author the experiment plan

Write `docs/exp/EXP-NNN-<slug>.md` from `docs/_templates/experiment.md`. Required sections (parser-enforced):

- `## Hypothesis`
- `## Method`
- `## Success / failure criteria`
- `## Kill criteria`

Recommended sections: Why this matters · Confounders to control · Pre-mortem · Estimated cost.

**Commit the plan to git BEFORE running anything.** The commit SHA is the pre-registration timestamp.

## 2. Pre-register

```bash
just exp-register docs/exp/EXP-NNN-<slug>.md
# or directly:
uv run lab exp register docs/exp/EXP-NNN-<slug>.md --hypothesis "<one-liner>"
```

If `lab exp validate` reports missing sections or a dirty git state, fix and re-commit before proceeding.

## 3. Author the sweep config

`conf/sweep/EXP-NNN-<slug>.yaml`. Mirror the slug in `experiment.slug` so the sweep attaches to the registered plan.

Sanity-check:

```bash
uv run lab sweep run conf/sweep/EXP-NNN-<slug>.yaml --dry-run
```

Verify cell count matches expectation (`models × configs × tasks × seeds`).

## 4. Run

```bash
uv run lab sweep run conf/sweep/EXP-NNN-<slug>.yaml --enforce-pre-registration
```

The runner:
- creates a manifest per cell, persisted to Postgres + MinIO
- streams traces as JSONL blobs to `s3://lab/runs/YYYY-MM/DD/<run_id>/`
- writes one `experiment_runs` row per cell (deterministic `run_id`)
- uses the Valkey GPU lease for local models (serial)
- is fully idempotent — re-running with `--resume` (default) skips done cells

**If the process dies mid-sweep**, just re-run the same command. Resume picks up exactly where it stopped.

## 5. Apply evaluators

```bash
uv run lab eval apply EXP-NNN-<slug> [--no-judge] [--only EVALUATOR_NAME]
```

`--no-judge` skips LLM-judge evaluators (conserves Ollama Cloud budget). Use `--only` to apply a single evaluator (e.g. while iterating on a new one).

## 6. Generate the report

```bash
uv run lab analyze report EXP-NNN-<slug> --out docs/findings/F-NNN-<slug>-report.md
```

## 7. Distill a finding (if warranted)

If the result is significant or surprising, write a proper finding:

```bash
uv run lab finding new F-NNN "one-line claim"
$EDITOR docs/findings/F-NNN-<slug>.md
uv run lab finding sync
```

Followed by `git commit` and `git push` if applicable.

## 8. Post-mortem (any sweep that took ≥ 1 hour or surprised us)

`docs/postmortems/PM-EXP-NNN.md` from the postmortem template. Record what we predicted, what happened, why the gap, what we'd do differently.

---

## Anti-patterns

- **Running a sweep without a pre-registered plan.** Use `--enforce-pre-registration` to make this impossible.
- **Single-seed claims.** Lab default is N≥8 for any number reported in a finding. See `protocols/reliability-sweep.md`.
- **Iterating on the evaluator AFTER seeing the per-run scores.** Define the evaluator + threshold in the plan; commit; then apply.
- **Discarding errored cells silently.** `error` rows must be investigated; they are signal.

## Quick reference

| Command | Purpose |
|---|---|
| `lab tasks load PATH` | register a task suite |
| `lab exp validate PATH` | dry-run plan validation |
| `lab exp register PATH` | pre-register the plan |
| `lab sweep run CONFIG [--enforce-pre-registration]` | run the sweep |
| `lab eval apply SLUG` | score all runs |
| `lab analyze report SLUG --out PATH` | markdown summary |
| `lab finding new F-NNN "claim"` | scaffold finding |
| `lab finding sync` | mirror findings/ to DB |
