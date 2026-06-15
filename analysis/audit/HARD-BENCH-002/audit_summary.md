# Trajectory audit — HARD-BENCH-002

96 scored episodes, 29 flags, 5 LLM audits.

## Flag counts per model

| model | episodes | narration | budget_exhausted | thrash | suspicious_pass |
| --- | --- | --- | --- | --- | --- |
| devstral-24b | 32 | 3 | 2 | 2 | 5 |
| gemma4-12b | 32 | 0 | 1 | 0 | 4 |
| qwen3-coder-30b | 32 | 4 | 2 | 1 | 5 |

## Flag details

- `narration` devstral-24b / code-topo-sort / s1: 0 structured tool calls across episode
- `narration` devstral-24b / shell-multifile-dedup-latest / s1: 0 structured tool calls across episode
- `narration` devstral-24b / shell-top-error-sources / s1: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-expr-parser-fix / s1: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-fibonacci-bug-fix / s1: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-interval-merge-fix / s1: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-topo-sort / s1: 0 structured tool calls across episode
- `budget_exhausted` devstral-24b / data-json-flatten-top-spender / s1: terminated_reason=max_turns_reached
- `budget_exhausted` devstral-24b / multi-etl-log-user-join / s1: terminated_reason=max_turns_reached
- `budget_exhausted` gemma4-12b / shell-fragment-reassembly / s1: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-coder-30b / shell-fragment-reassembly / s1: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-coder-30b / shell-multifile-dedup-latest / s1: terminated_reason=max_turns_reached
- `thrash` devstral-24b / data-json-flatten-top-spender / s1: 8 calls vs passing median 3 (>1.5x), still failed
- `thrash` devstral-24b / multi-etl-log-user-join / s1: 13 calls vs passing median 4.5 (>1.5x), still failed
- `thrash` qwen3-coder-30b / shell-access-log-slow-error-endpoints / s1: 6 calls vs passing median 3 (>1.5x), still failed
- `suspicious_pass` gemma4-12b / code-dijkstra-multifile / s1: passed with 3 calls; other passers' min is 8
- `suspicious_pass` devstral-24b / code-fibonacci-bug-fix / s1: passed with 5 calls; other passers' min is 8
- `suspicious_pass` devstral-24b / code-lru-cache-trace / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` qwen3-coder-30b / data-csv-join-revenue-by-region / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` gemma4-12b / data-duplicate-transaction-count / s1: passed with 3 calls; other passers' min is 6
- `suspicious_pass` qwen3-coder-30b / data-duplicate-transaction-count / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` devstral-24b / data-jsonl-session-median-duration / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` qwen3-coder-30b / data-log-parse-p95-latency / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` devstral-24b / data-multi-file-weighted-score / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` gemma4-12b / multi-http-catalog-pricing / s1: passed with 9 calls; other passers' min is 14
- `suspicious_pass` qwen3-coder-30b / multi-http-catalog-pricing / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` gemma4-12b / shell-multifile-dedup-latest / s1: passed with 6 calls; other passers' min is 10
- `suspicious_pass` qwen3-coder-30b / shell-sensitive-config-keys / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` devstral-24b / shell-tsv-dedup-median-by-dept / s1: passed with 2 calls; other passers' min is 5

## LLM verdicts

| model | task | seed | flag | verdict | rationale |
| --- | --- | --- | --- | --- | --- |
| gemma4-12b | code-dijkstra-multifile | 1 | suspicious_pass | shortcut | The agent hardcoded the edge list directly into the Python code instead of programmatically parsing the output of fs_grep, bypassing the required parsing step and achieving a pass in only 3 tool calls |
| devstral-24b | code-fibonacci-bug-fix | 1 | suspicious_pass | genuine | The agent followed all required steps: grepped for the file, read it, fixed the base-case bug, ran the script to verify, and wrote the result to answer.txt. The low call count reflects efficiency, not |
| devstral-24b | code-lru-cache-trace | 1 | suspicious_pass | genuine | The agent read the operations file, implemented an LRU cache using OrderedDict in python_eval, replayed the operations, and produced the correct comma-separated results. Although it wrote the answer f |
| qwen3-coder-30b | data-csv-join-revenue-by-region | 1 | suspicious_pass | genuine | The agent read both CSV files, used python_eval to join, filter completed orders, sum by region, and write the result to the output file, then verified the result by reading it back. The mechanical fl |
| gemma4-12b | data-duplicate-transaction-count | 1 | suspicious_pass | genuine | The agent read the CSV, wrote a correct Python script using Counter to find distinct txn_id values appearing more than once, and wrote the result. The low call count reflects efficient use of a single |
