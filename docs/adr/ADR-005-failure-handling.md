# ADR-005: Failure handling — harness errors vs model errors

Status: accepted
Date: 2026-05-25
Deciders: Matt Rachlin

## Context

A sweep cell can "fail" in two very different ways:

- **Harness error** — the lab's infrastructure broke. Ollama returned a 500. The DB connection dropped. The trace upload failed. The model failed to load. These tell us nothing about the model's capability; they're operational debt.
- **Model error** — the model produced an empty response, a malformed answer, a refusal, or just the wrong answer. These ARE signal — they're what the eval is supposed to measure.

If we conflate the two, two bad things happen:
1. Aggregate pass rates are artificially low because of infra failures.
2. Real model failures are dismissed as "probably infra" and never investigated.

The Phase 2 RELIABILITY-001 sweep had a mid-sweep process death (104/120 done; resumed cleanly). At handoff, the F-002 finding included an empty-response count for qwen3-14b that needed careful interpretation — was it a model behavior or a budget squeeze?

We need a clear rule.

## Decision

**Three distinct `experiment_runs.status` values, with explicit semantics:**

| status | Meaning | Counts toward eval? |
|---|---|---|
| `done` | The model returned a response (any content, including empty) without infra error | YES — every evaluator sees this row |
| `error` | Infrastructure failed during the call (HTTP 5xx, timeout, model-load failure, MinIO write error) | NO — excluded from eval_results aggregates; investigated separately |
| `killed` | Process received SIGTERM mid-call (manual abort, OOM kill, sweep interruption) | NO — counted as "interrupted, not attempted" |

Empty response, wrong answer, refusal — all of these are `done`. They are model behaviors. Evaluators score them as 0.0 (or `not_empty` flags them) but the row IS a real evaluation.

The runner catches all per-cell exceptions and routes them: connection errors, HTTP 5xx, model-load failures, parse failures all become `error`. SIGTERM during a cell becomes `killed`.

`error` and `killed` rows still get a `manifest_sha` recorded, so we know which sweep instance produced them. Trace paths may be null.

When an EXP plan defines a success criterion, it's stated as a function of `done` rows only:

> Pass criterion: pass@1 ≥ 0.7 across N=8 done runs on PBS-math tasks.

Not: "across N=8 attempts." If 1 of 8 was an `error`, the cell is re-run (resumable) and the criterion evaluated on the corrected 8 done rows.

## Consequences

- **Easier**: aggregates are clean. Pass rates measure model behavior, not infrastructure stability.
- **Harder**: investigating `error` rows is a separate workflow. The lab needs a `lab sweep errors EXP-SLUG` command (Phase 3.8 TODO) that surfaces them.
- **Risks**: a model that systematically times out could be silently dropped from the comparison. Mitigation: report `error_rate` alongside pass rates in the analyze report. If `error_rate > 5%` for a (model, config) pair, the plan's success criterion is flagged as unstable.

## Considered alternatives

- **Score `error` as a failure** ("the model is responsible for not crashing the runner"). Rejected — a 500 from Ollama tells us about Ollama, not about the model.
- **Drop `error` rows silently**. Rejected — we lose the signal that something needs operational attention.
- **Single bucketed `failed` status**. Rejected for the reasons in Context.
