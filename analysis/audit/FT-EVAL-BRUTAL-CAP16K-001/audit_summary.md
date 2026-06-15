# Trajectory audit — FT-EVAL-BRUTAL-CAP16K-001

72 scored episodes, 15 flags, 0 LLM audits.

## Flag counts per model

| model | episodes | narration | budget_exhausted | error_loop | thrash |
| --- | --- | --- | --- | --- | --- |
| qwen3-4b-ft-toolcall-q4-latest | 72 | 5 | 8 | 1 | 1 |

## Flag details

- `narration` qwen3-4b-ft-toolcall-q4-latest / spec-invoice-window / s2: 0 structured tool calls across episode
- `narration` qwen3-4b-ft-toolcall-q4-latest / spec-invoice-window / s3: 0 structured tool calls across episode
- `narration` qwen3-4b-ft-toolcall-q4-latest / spec-tournament-podium / s1: 0 structured tool calls across episode
- `narration` qwen3-4b-ft-toolcall-q4-latest / spec-tournament-podium / s2: 0 structured tool calls across episode
- `narration` qwen3-4b-ft-toolcall-q4-latest / spec-tournament-podium / s3: 0 structured tool calls across episode
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / debug-job-state-machine / s1: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / debug-job-state-machine / s2: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / debug-job-state-machine / s3: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / longhaul-orders-etl-staged / s3: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / recovery-moved-records / s1: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / recovery-truncated-json / s1: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / recovery-truncated-json / s3: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / spec-invoice-window / s1: terminated_reason=max_turns_reached
- `error_loop` qwen3-4b-ft-toolcall-q4-latest / recovery-decoy-metrics / s1: 4 consecutive all-error turns
- `thrash` qwen3-4b-ft-toolcall-q4-latest / recovery-moved-records / s1: 14 calls vs passing median 5 (>1.5x), still failed

## LLM verdicts

Not run (pass --llm-audit to audit flagged episodes).
