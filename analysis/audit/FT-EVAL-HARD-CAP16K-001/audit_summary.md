# Trajectory audit — FT-EVAL-HARD-CAP16K-001

96 scored episodes, 23 flags, 0 LLM audits.

## Flag counts per model

| model | episodes | narration | budget_exhausted | error_loop | thrash |
| --- | --- | --- | --- | --- | --- |
| qwen3-4b-ft-toolcall-q4-latest | 96 | 4 | 10 | 2 | 7 |

## Flag details

- `narration` qwen3-4b-ft-toolcall-q4-latest / code-lru-cache-trace / s1: 0 structured tool calls across episode
- `narration` qwen3-4b-ft-toolcall-q4-latest / code-lru-cache-trace / s2: 0 structured tool calls across episode
- `narration` qwen3-4b-ft-toolcall-q4-latest / code-lru-cache-trace / s3: 0 structured tool calls across episode
- `narration` qwen3-4b-ft-toolcall-q4-latest / data-payment-reconciliation / s3: 0 structured tool calls across episode
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / code-expr-parser-fix / s2: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / code-expr-parser-fix / s3: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / code-fibonacci-bug-fix / s2: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / code-fibonacci-bug-fix / s3: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / data-duplicate-transaction-count / s2: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / data-duplicate-transaction-count / s3: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / data-multi-file-weighted-score / s2: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / multi-config-driven-transform / s2: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / shell-top-error-sources / s2: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / shell-top-error-sources / s3: terminated_reason=max_turns_reached
- `error_loop` qwen3-4b-ft-toolcall-q4-latest / code-expr-parser-fix / s2: 4 consecutive all-error turns
- `error_loop` qwen3-4b-ft-toolcall-q4-latest / code-expr-parser-fix / s3: 4 consecutive all-error turns
- `thrash` qwen3-4b-ft-toolcall-q4-latest / code-fibonacci-bug-fix / s2: 10 calls vs passing median 5 (>1.5x), still failed
- `thrash` qwen3-4b-ft-toolcall-q4-latest / code-fibonacci-bug-fix / s3: 10 calls vs passing median 5 (>1.5x), still failed
- `thrash` qwen3-4b-ft-toolcall-q4-latest / data-duplicate-transaction-count / s2: 8 calls vs passing median 3 (>1.5x), still failed
- `thrash` qwen3-4b-ft-toolcall-q4-latest / data-duplicate-transaction-count / s3: 8 calls vs passing median 3 (>1.5x), still failed
- `thrash` qwen3-4b-ft-toolcall-q4-latest / data-multi-file-weighted-score / s2: 14 calls vs passing median 4 (>1.5x), still failed
- `thrash` qwen3-4b-ft-toolcall-q4-latest / multi-config-driven-transform / s2: 12 calls vs passing median 3 (>1.5x), still failed
- `thrash` qwen3-4b-ft-toolcall-q4-latest / shell-fragment-reassembly / s1: 9 calls vs passing median 1 (>1.5x), still failed

## LLM verdicts

Not run (pass --llm-audit to audit flagged episodes).
