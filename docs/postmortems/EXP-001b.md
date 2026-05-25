# Postmortem: EXP-001b — qwen3 reasoning + max_tokens ablation

Date: 2026-05-25
Experiment: EXP-001b
Finding: [F-004](../findings/F-004-qwen3-reasoning-ablation.md)

## What happened

Ran a focused 384-cell ablation on qwen3-14b-q4 to isolate the F-003 H3 confound. Three pre-registered hypotheses, all refuted — including one (H3) that flipped sign. Total wall time 2 h 50 min (predicted 30 min). Zero cell errors, zero kill criteria fired.

## What went well

- **Smoke test caught a real infra trap before the full sweep.** The smoke run showed `/no_think` as a prompt token doesn't actually disable qwen3 thinking on Ollama — only the API-level `think: false` parameter does. Fixed by extending the sweep runner to forward `config.extra.*` keys, which keeps experiment specs declarative.
- **Pre-registration discipline held under temptation.** When the rate dropped to 1 cell/min on B.1 (vs 20/min on B.2), I was tempted to kill+restart with N=4 or drop B.1 entirely. The user explicitly chose "wait it out" instead of compromising the design. The verdicts came in clean.
- **The sweep runner's slug regex bug (lowercase suffix `b`) was caught at registration time** by the EXP-001b plan failing to register correctly — the validator misread the slug as `EXP-001` and overwrote that row. Fixed `SLUG_RE` to allow lowercase tail; restored EXP-001's metadata. Good lesson: surfaceable validators beat silent ones.

## What went wrong

### 1. The wall-time estimate was 3× too low (medium)

I estimated 30 min, actual was 170 min. Root cause: I assumed qwen3 with 2048-token budget would use ~1.5× the time of 1024-budget. Actual was ~10× — qwen3 with reasoning will use *all* of any budget you give it, and the extra tokens are sequential. Some single cells took >90 s.

**Action queued**: Add a per-cell wall-clock timeout to sweep configs (separate from request_timeout). For reasoning-by-default models, default it to ~30 s and surface skipped cells in the eval report rather than blocking the whole sweep.

### 2. Slug regex bug bit on first lowercase-suffix slug (low; fixed)

`SLUG_RE = r"^[A-Z][A-Z0-9_-]{2,63}$"` rejected `EXP-001b`. The validator fell back to the filename regex which truncated to `EXP-001`, and the registration UPSERTed onto EXP-001's row, clobbering it. Caught immediately because the printed message said "registered EXP-001 (sha=…)" with a sha we didn't expect.

**Fix shipped**: Slug regex now allows lowercase tail (`[A-Za-z0-9_-]`), and the filename fallback allows an optional trailing lowercase letter (`-\d+[a-z]?`). EXP-001's row was re-registered with the correct SHA. Tests still pass.

### 3. `/no_think` prompt token does not actually disable qwen3 thinking (medium; documented)

This is a model-behavior trap, not our bug — but it would have silently corrupted the experiment if the smoke step had been skipped. The "obvious" approach (system prompt or user-message suffix) leaves the thinking channel fully active. Only the API-level `think: false` parameter disables it.

**Action queued**: Add a checklist item to `sops/sweep-execution.md`: for any reasoning-by-default model, smoke-test the disable-mechanism before the full sweep and document completion-token deltas.

### 4. The Welch p-values barely cross significance on big-looking deltas (informational)

H3's "drop" of −28 pp on math had p = 0.276. The effect size is huge but n=8 tasks per category means tight CIs are hard. F-004 reports this honestly but for any future experiment where claims need to be statistically defensible at α=0.05, target N≥16 tasks per category (curating new PBS tasks is the long pole here, not compute).

## What I'd do differently

- **Test the rate hypothesis before scoping**: before committing to the 384-cell EXP-001b, run 4-cell smoke at each config and extrapolate. I had the smoke harness; I just didn't extrapolate latency to wall-time.
- **For reasoning-on configs, default to 4096 max_tokens with a hard per-cell timeout.** The constraint isn't tokens, it's wall time. Cap wall time and let cells skip rather than blocking the matrix.
- **Pre-register the metric direction explicitly**: H3's prediction said "drop ≥ 10pp" — but "drop" is ambiguous about sign. The analyzer flagged the inversion (drop = −0.281), which is the right behavior, but a more careful pre-reg would have written `(baseline − B.2) ≥ +0.10` to remove the ambiguity. Minor.

## Action items

| # | Item | Owner | Status |
|---|---|---|---|
| 1 | Per-cell wall-time cap in sweep config (`max_wall_sec`?) | next session | open |
| 2 | SOP update: smoke-test reasoning-disable mechanism + completion-token deltas | next session | open |
| 3 | Slug regex fix shipped (commit ae3f1db3) | this session | done |
| 4 | Sweep runner forwards `config.extra.*` to request body | this session | done |
| 5 | EXP-001c plan: agent-loop / tool-use scaffold — does reasoning earn its keep there? | follow-on | open |
| 6 | Update lab default qwen3 invocation to `think: false` | next session | open |

## Files

- Finding: [F-004-qwen3-reasoning-ablation](../findings/F-004-qwen3-reasoning-ablation.md)
- Plan: [EXP-001b](../exp/EXP-001b.md)
- Sweep config: [`conf/sweep/EXP-001b.yaml`](../../conf/sweep/EXP-001b.yaml)
