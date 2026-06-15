# Trajectory audit — ARCH-BENCH-001

192 scored episodes, 22 flags, 3 LLM audits.

## Flag counts per model

| model | episodes | narration | budget_exhausted | error_loop | thrash |
| --- | --- | --- | --- | --- | --- |
| gpt-oss-20b-local | 96 | 7 | 3 | 0 | 0 |
| granite4-tiny-h | 96 | 3 | 3 | 3 | 3 |

## Flag details

- `narration` gpt-oss-20b-local / code-dijkstra-multifile / s1: 0 structured tool calls across episode
- `narration` gpt-oss-20b-local / code-dijkstra-multifile / s2: 0 structured tool calls across episode
- `narration` gpt-oss-20b-local / code-dijkstra-multifile / s3: 0 structured tool calls across episode
- `narration` gpt-oss-20b-local / code-expr-parser-fix / s1: 0 structured tool calls across episode
- `narration` gpt-oss-20b-local / code-expr-parser-fix / s2: 0 structured tool calls across episode
- `narration` gpt-oss-20b-local / code-expr-parser-fix / s3: 0 structured tool calls across episode
- `narration` gpt-oss-20b-local / code-fibonacci-bug-fix / s1: 0 structured tool calls across episode
- `narration` granite4-tiny-h / shell-top-error-sources / s1: 0 structured tool calls across episode
- `narration` granite4-tiny-h / shell-top-error-sources / s2: 0 structured tool calls across episode
- `narration` granite4-tiny-h / shell-top-error-sources / s3: 0 structured tool calls across episode
- `budget_exhausted` gpt-oss-20b-local / shell-fragment-reassembly / s1: terminated_reason=max_turns_reached
- `budget_exhausted` gpt-oss-20b-local / shell-fragment-reassembly / s2: terminated_reason=max_turns_reached
- `budget_exhausted` gpt-oss-20b-local / shell-fragment-reassembly / s3: terminated_reason=max_turns_reached
- `budget_exhausted` granite4-tiny-h / shell-fragment-reassembly / s1: terminated_reason=max_turns_reached
- `budget_exhausted` granite4-tiny-h / shell-fragment-reassembly / s2: terminated_reason=max_turns_reached
- `budget_exhausted` granite4-tiny-h / shell-fragment-reassembly / s3: terminated_reason=max_turns_reached
- `error_loop` granite4-tiny-h / shell-fragment-reassembly / s1: 3 consecutive all-error turns
- `error_loop` granite4-tiny-h / shell-fragment-reassembly / s2: 3 consecutive all-error turns
- `error_loop` granite4-tiny-h / shell-fragment-reassembly / s3: 3 consecutive all-error turns
- `thrash` granite4-tiny-h / shell-tsv-dedup-median-by-dept / s1: 6 calls vs passing median 3 (>1.5x), still failed
- `thrash` granite4-tiny-h / shell-tsv-dedup-median-by-dept / s2: 6 calls vs passing median 3 (>1.5x), still failed
- `thrash` granite4-tiny-h / shell-tsv-dedup-median-by-dept / s3: 6 calls vs passing median 3 (>1.5x), still failed

## LLM verdicts

| model | task | seed | flag | verdict | rationale |
| --- | --- | --- | --- | --- | --- |
| granite4-tiny-h | shell-tsv-dedup-median-by-dept | 1 | thrash | lucky_fail | The agent failed with score 0 after 6 thrashing attempts of truncated Python scripts that never properly completed the pipeline; there is no luck here since it scored zero, but the verdict options con |
| granite4-tiny-h | shell-tsv-dedup-median-by-dept | 2 | thrash | lucky_fail | The agent repeatedly ran truncated Python scripts that likely produced errors or incorrect output, and the final score was 0 (FAIL), but the agent falsely claimed success; no shortcut was exploited, j |
| granite4-tiny-h | shell-tsv-dedup-median-by-dept | 3 | thrash | genuine | The agent made a legitimate multi-attempt effort to parse, deduplicate, and compute medians using Python, but ultimately failed (score 0) — no shortcut or luck involved. |
