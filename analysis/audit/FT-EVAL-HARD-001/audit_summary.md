# Trajectory audit — FT-EVAL-HARD-001

192 scored episodes, 54 flags, 0 LLM audits.

## Flag counts per model

| model | episodes | narration | budget_exhausted | error_loop | thrash | suspicious_pass |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3-4b | 96 | 21 | 5 | 0 | 11 | 0 |
| qwen3-4b-ft-toolcall-q4-latest | 96 | 5 | 6 | 2 | 3 | 1 |

## Flag details

- `narration` qwen3-4b / code-expr-parser-fix / s1: 0 structured tool calls across episode
- `narration` qwen3-4b / code-lru-cache-trace / s1: 0 structured tool calls across episode
- `narration` qwen3-4b / code-lru-cache-trace / s2: 0 structured tool calls across episode
- `narration` qwen3-4b / code-lru-cache-trace / s3: 0 structured tool calls across episode
- `narration` qwen3-4b / code-topo-sort / s1: 0 structured tool calls across episode
- `narration` qwen3-4b / code-topo-sort / s2: 0 structured tool calls across episode
- `narration` qwen3-4b / code-topo-sort / s3: 0 structured tool calls across episode
- `narration` qwen3-4b / data-jsonl-session-median-duration / s1: 0 structured tool calls across episode
- `narration` qwen3-4b / data-jsonl-session-median-duration / s2: 0 structured tool calls across episode
- `narration` qwen3-4b / data-multi-file-weighted-score / s1: 0 structured tool calls across episode
- `narration` qwen3-4b / data-multi-file-weighted-score / s2: 0 structured tool calls across episode
- `narration` qwen3-4b / data-multi-file-weighted-score / s3: 0 structured tool calls across episode
- `narration` qwen3-4b / shell-dept-salary-sum / s1: 0 structured tool calls across episode
- `narration` qwen3-4b / shell-dept-salary-sum / s2: 0 structured tool calls across episode
- `narration` qwen3-4b / shell-dept-salary-sum / s3: 0 structured tool calls across episode
- `narration` qwen3-4b / shell-multifile-dedup-latest / s1: 0 structured tool calls across episode
- `narration` qwen3-4b / shell-multifile-dedup-latest / s2: 0 structured tool calls across episode
- `narration` qwen3-4b / shell-multifile-dedup-latest / s3: 0 structured tool calls across episode
- `narration` qwen3-4b / shell-top-error-sources / s1: 0 structured tool calls across episode
- `narration` qwen3-4b / shell-top-error-sources / s2: 0 structured tool calls across episode
- `narration` qwen3-4b / shell-top-error-sources / s3: 0 structured tool calls across episode
- `narration` qwen3-4b-ft-toolcall-q4-latest / data-log-parse-p95-latency / s2: 0 structured tool calls across episode
- `narration` qwen3-4b-ft-toolcall-q4-latest / data-log-parse-p95-latency / s3: 0 structured tool calls across episode
- `narration` qwen3-4b-ft-toolcall-q4-latest / data-payment-reconciliation / s1: 0 structured tool calls across episode
- `narration` qwen3-4b-ft-toolcall-q4-latest / data-payment-reconciliation / s2: 0 structured tool calls across episode
- `narration` qwen3-4b-ft-toolcall-q4-latest / data-payment-reconciliation / s3: 0 structured tool calls across episode
- `budget_exhausted` qwen3-4b / data-csv-join-revenue-by-region / s1: terminated_reason=budget_exhausted
- `budget_exhausted` qwen3-4b / data-csv-join-revenue-by-region / s2: terminated_reason=budget_exhausted
- `budget_exhausted` qwen3-4b / data-csv-join-revenue-by-region / s3: terminated_reason=budget_exhausted
- `budget_exhausted` qwen3-4b / shell-fragment-reassembly / s1: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b / shell-fragment-reassembly / s3: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / code-expr-parser-fix / s1: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / code-expr-parser-fix / s3: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / data-json-flatten-top-spender / s1: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / multi-config-driven-transform / s2: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / shell-top-error-sources / s2: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-4b-ft-toolcall-q4-latest / shell-top-error-sources / s3: terminated_reason=max_turns_reached
- `error_loop` qwen3-4b-ft-toolcall-q4-latest / code-expr-parser-fix / s1: 3 consecutive all-error turns
- `error_loop` qwen3-4b-ft-toolcall-q4-latest / code-expr-parser-fix / s3: 4 consecutive all-error turns
- `thrash` qwen3-4b / code-interval-merge-fix / s3: 12 calls vs passing median 4 (>1.5x), still failed
- `thrash` qwen3-4b / data-inventory-turnover / s2: 2 calls vs passing median 1 (>1.5x), still failed
- `thrash` qwen3-4b / data-inventory-turnover / s3: 2 calls vs passing median 1 (>1.5x), still failed
- `thrash` qwen3-4b / data-json-flatten-top-spender / s1: 3 calls vs passing median 1.5 (>1.5x), still failed
- `thrash` qwen3-4b-ft-toolcall-q4-latest / data-json-flatten-top-spender / s1: 8 calls vs passing median 1.5 (>1.5x), still failed
- `thrash` qwen3-4b / data-log-parse-p95-latency / s1: 9 calls vs passing median 1 (>1.5x), still failed
- `thrash` qwen3-4b / data-log-parse-p95-latency / s2: 2 calls vs passing median 1 (>1.5x), still failed
- `thrash` qwen3-4b / data-log-parse-p95-latency / s3: 2 calls vs passing median 1 (>1.5x), still failed
- `thrash` qwen3-4b-ft-toolcall-q4-latest / multi-config-driven-transform / s2: 12 calls vs passing median 3 (>1.5x), still failed
- `thrash` qwen3-4b / shell-fragment-reassembly / s1: 13 calls vs passing median 1 (>1.5x), still failed
- `thrash` qwen3-4b / shell-fragment-reassembly / s2: 8 calls vs passing median 1 (>1.5x), still failed
- `thrash` qwen3-4b / shell-fragment-reassembly / s3: 12 calls vs passing median 1 (>1.5x), still failed
- `thrash` qwen3-4b-ft-toolcall-q4-latest / shell-fragment-reassembly / s1: 9 calls vs passing median 1 (>1.5x), still failed
- `thrash` qwen3-4b / shell-sensitive-config-keys / s2: 6 calls vs passing median 2 (>1.5x), still failed
- `suspicious_pass` qwen3-4b-ft-toolcall-q4-latest / data-multi-file-weighted-score / s3: passed with 2 calls; other passers' min is 5

## LLM verdicts

Not run (pass --llm-audit to audit flagged episodes).
