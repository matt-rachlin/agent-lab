# Trajectory audit — HARD-BENCH-003

768 scored episodes, 98 flags, 20 LLM audits.

## Flag counts per model

| model | episodes | narration | budget_exhausted | thrash | suspicious_pass |
| --- | --- | --- | --- | --- | --- |
| devstral-24b | 256 | 23 | 10 | 20 | 0 |
| gemma4-12b | 256 | 3 | 1 | 1 | 1 |
| qwen3-coder-30b | 256 | 32 | 4 | 0 | 3 |

## Flag details

- `narration` devstral-24b / code-topo-sort / s1: 0 structured tool calls across episode
- `narration` devstral-24b / code-topo-sort / s2: 0 structured tool calls across episode
- `narration` devstral-24b / code-topo-sort / s3: 0 structured tool calls across episode
- `narration` devstral-24b / code-topo-sort / s4: 0 structured tool calls across episode
- `narration` devstral-24b / code-topo-sort / s5: 0 structured tool calls across episode
- `narration` devstral-24b / code-topo-sort / s6: 0 structured tool calls across episode
- `narration` devstral-24b / code-topo-sort / s7: 0 structured tool calls across episode
- `narration` devstral-24b / code-topo-sort / s8: 0 structured tool calls across episode
- `narration` devstral-24b / shell-multifile-dedup-latest / s2: 0 structured tool calls across episode
- `narration` devstral-24b / shell-multifile-dedup-latest / s3: 0 structured tool calls across episode
- `narration` devstral-24b / shell-multifile-dedup-latest / s4: 0 structured tool calls across episode
- `narration` devstral-24b / shell-multifile-dedup-latest / s5: 0 structured tool calls across episode
- `narration` devstral-24b / shell-multifile-dedup-latest / s6: 0 structured tool calls across episode
- `narration` devstral-24b / shell-multifile-dedup-latest / s7: 0 structured tool calls across episode
- `narration` devstral-24b / shell-multifile-dedup-latest / s8: 0 structured tool calls across episode
- `narration` devstral-24b / shell-top-error-sources / s1: 0 structured tool calls across episode
- `narration` devstral-24b / shell-top-error-sources / s2: 0 structured tool calls across episode
- `narration` devstral-24b / shell-top-error-sources / s3: 0 structured tool calls across episode
- `narration` devstral-24b / shell-top-error-sources / s4: 0 structured tool calls across episode
- `narration` devstral-24b / shell-top-error-sources / s5: 0 structured tool calls across episode
- `narration` devstral-24b / shell-top-error-sources / s6: 0 structured tool calls across episode
- `narration` devstral-24b / shell-top-error-sources / s7: 0 structured tool calls across episode
- `narration` devstral-24b / shell-top-error-sources / s8: 0 structured tool calls across episode
- `narration` gemma4-12b / shell-top-error-sources / s1: 0 structured tool calls across episode
- `narration` gemma4-12b / shell-top-error-sources / s2: 0 structured tool calls across episode
- `narration` gemma4-12b / shell-top-error-sources / s4: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-expr-parser-fix / s1: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-expr-parser-fix / s2: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-expr-parser-fix / s3: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-expr-parser-fix / s4: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-expr-parser-fix / s5: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-expr-parser-fix / s6: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-expr-parser-fix / s7: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-expr-parser-fix / s8: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-fibonacci-bug-fix / s1: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-fibonacci-bug-fix / s2: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-fibonacci-bug-fix / s3: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-fibonacci-bug-fix / s4: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-fibonacci-bug-fix / s5: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-fibonacci-bug-fix / s6: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-fibonacci-bug-fix / s7: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-fibonacci-bug-fix / s8: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-interval-merge-fix / s1: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-interval-merge-fix / s2: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-interval-merge-fix / s3: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-interval-merge-fix / s4: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-interval-merge-fix / s5: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-interval-merge-fix / s6: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-interval-merge-fix / s7: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-interval-merge-fix / s8: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-topo-sort / s1: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-topo-sort / s2: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-topo-sort / s3: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-topo-sort / s4: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-topo-sort / s5: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-topo-sort / s6: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-topo-sort / s7: 0 structured tool calls across episode
- `narration` qwen3-coder-30b / code-topo-sort / s8: 0 structured tool calls across episode
- `budget_exhausted` devstral-24b / code-sample-variance-fix / s2: terminated_reason=max_turns_reached
- `budget_exhausted` devstral-24b / code-sample-variance-fix / s3: terminated_reason=max_turns_reached
- `budget_exhausted` devstral-24b / code-sample-variance-fix / s4: terminated_reason=max_turns_reached
- `budget_exhausted` devstral-24b / code-sample-variance-fix / s5: terminated_reason=max_turns_reached
- `budget_exhausted` devstral-24b / code-sample-variance-fix / s6: terminated_reason=max_turns_reached
- `budget_exhausted` devstral-24b / code-sample-variance-fix / s8: terminated_reason=max_turns_reached
- `budget_exhausted` devstral-24b / multi-config-driven-transform / s1: terminated_reason=max_turns_reached
- `budget_exhausted` devstral-24b / multi-config-driven-transform / s3: terminated_reason=max_turns_reached
- `budget_exhausted` devstral-24b / multi-config-driven-transform / s5: terminated_reason=max_turns_reached
- `budget_exhausted` devstral-24b / multi-config-driven-transform / s6: terminated_reason=max_turns_reached
- `budget_exhausted` gemma4-12b / code-interval-merge-fix / s1: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-coder-30b / code-dijkstra-multifile / s6: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-coder-30b / data-log-parse-p95-latency / s1: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-coder-30b / shell-multifile-dedup-latest / s2: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-coder-30b / shell-multifile-dedup-latest / s3: terminated_reason=max_turns_reached
- `thrash` gemma4-12b / code-interval-merge-fix / s1: 8 calls vs passing median 4 (>1.5x), still failed
- `thrash` devstral-24b / code-sample-variance-fix / s2: 12 calls vs passing median 7 (>1.5x), still failed
- `thrash` devstral-24b / code-sample-variance-fix / s3: 12 calls vs passing median 7 (>1.5x), still failed
- `thrash` devstral-24b / code-sample-variance-fix / s4: 12 calls vs passing median 7 (>1.5x), still failed
- `thrash` devstral-24b / code-sample-variance-fix / s5: 12 calls vs passing median 7 (>1.5x), still failed
- `thrash` devstral-24b / code-sample-variance-fix / s6: 12 calls vs passing median 7 (>1.5x), still failed
- `thrash` devstral-24b / code-sample-variance-fix / s8: 12 calls vs passing median 7 (>1.5x), still failed
- `thrash` devstral-24b / data-json-flatten-top-spender / s2: 6 calls vs passing median 3 (>1.5x), still failed
- `thrash` devstral-24b / data-json-flatten-top-spender / s3: 6 calls vs passing median 3 (>1.5x), still failed
- `thrash` devstral-24b / data-json-flatten-top-spender / s4: 6 calls vs passing median 3 (>1.5x), still failed
- `thrash` devstral-24b / data-json-flatten-top-spender / s5: 6 calls vs passing median 3 (>1.5x), still failed
- `thrash` devstral-24b / data-json-flatten-top-spender / s6: 6 calls vs passing median 3 (>1.5x), still failed
- `thrash` devstral-24b / data-json-flatten-top-spender / s7: 6 calls vs passing median 3 (>1.5x), still failed
- `thrash` devstral-24b / data-json-flatten-top-spender / s8: 6 calls vs passing median 3 (>1.5x), still failed
- `thrash` devstral-24b / data-log-parse-p95-latency / s1: 6 calls vs passing median 3 (>1.5x), still failed
- `thrash` devstral-24b / multi-config-driven-transform / s1: 12 calls vs passing median 4 (>1.5x), still failed
- `thrash` devstral-24b / multi-config-driven-transform / s3: 12 calls vs passing median 4 (>1.5x), still failed
- `thrash` devstral-24b / multi-config-driven-transform / s5: 12 calls vs passing median 4 (>1.5x), still failed
- `thrash` devstral-24b / multi-config-driven-transform / s6: 12 calls vs passing median 4 (>1.5x), still failed
- `thrash` devstral-24b / multi-grep-weighted-todo / s3: 8 calls vs passing median 4 (>1.5x), still failed
- `thrash` devstral-24b / multi-grep-weighted-todo / s7: 8 calls vs passing median 4 (>1.5x), still failed
- `suspicious_pass` qwen3-coder-30b / data-duplicate-transaction-count / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` qwen3-coder-30b / data-json-flatten-top-spender / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` qwen3-coder-30b / shell-sensitive-config-keys / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` gemma4-12b / shell-tsv-dedup-median-by-dept / s1: passed with zero fs_write/shell_exec calls; every other passer used them

## LLM verdicts

| model | task | seed | flag | verdict | rationale |
| --- | --- | --- | --- | --- | --- |
| gemma4-12b | code-interval-merge-fix | 1 | thrash | lucky_fail | The agent identified and fixed the bug correctly and computed the right answer, but never wrote it to 'answer.txt', running out of turns while re-reading the source file. It failed the task outright ( |
| devstral-24b | code-sample-variance-fix | 2 | thrash | lucky_fail | The agent correctly fixed the variance formula but never wrote the result to 'answer.txt' as required, instead thrashing on repeated python_eval calls until max turns. It failed the task outright (sco |
| devstral-24b | code-sample-variance-fix | 3 | thrash | lucky_fail | The agent correctly fixed the variance bug and computed the result, but never wrote it to 'answer.txt' as required, instead thrashing by repeatedly evaluating the same Python code until max turns were |
| devstral-24b | code-sample-variance-fix | 4 | thrash | lucky_fail | The agent correctly identified and fixed the variance bug, but got stuck repeatedly calling python_eval without ever writing the result to answer.txt, ultimately failing the task (score 0). Since it f |
| devstral-24b | code-sample-variance-fix | 5 | thrash | lucky_fail | The agent correctly fixed the variance formula and computed the result, but never wrote it to 'answer.txt' as required, instead thrashing by repeatedly calling python_eval without progressing to the f |
| devstral-24b | code-sample-variance-fix | 6 | thrash | lucky_fail | The agent correctly fixed the variance bug and computed the result multiple times, but never wrote it to 'answer.txt' as required, resulting in a score of 0. The agent failed due to a missing final st |
| devstral-24b | code-sample-variance-fix | 8 | thrash | lucky_fail | The agent correctly fixed the variance formula and computed the result, but never wrote it to 'answer.txt' as required, instead thrashing by repeatedly re-evaluating the same code until max turns were |
| qwen3-coder-30b | data-duplicate-transaction-count | 1 | suspicious_pass | genuine | The agent read the CSV file, wrote Python code using csv.DictReader and collections.defaultdict to count duplicate txn_ids, and verified the output by reading result.txt. The lack of fs_write/shell_ex |
| devstral-24b | data-json-flatten-top-spender | 2 | thrash | lucky_fail | The agent failed to write the result to /workspace/result.txt as required, only stating 'Eve' in conversation; it scored 0 and never completed the core task of writing the answer to the file. |
| devstral-24b | data-json-flatten-top-spender | 3 | thrash | genuine | The agent legitimately attempted to solve the task by reading the JSON data and running Python computations across multiple turns, but failed to write the result to /workspace/result.txt (only read th |
| devstral-24b | data-json-flatten-top-spender | 4 | thrash | lucky_fail | The agent failed to write the result to /workspace/result.txt (only read the file, never wrote), and the final score was 0. The agent's answer 'Eve' may or may not be correct, but it never completed t |
| devstral-24b | data-json-flatten-top-spender | 5 | thrash | lucky_fail | The agent computed the correct answer ('Eve') but never wrote it to /workspace/result.txt as required, resulting in a score of 0. It had the right answer but failed on the mechanical output step. |
| devstral-24b | data-json-flatten-top-spender | 6 | thrash | lucky_fail | The agent failed to write the result to /workspace/result.txt and instead just output 'Eve' in conversation; the final score is 0 (FAIL), so it didn't luck out either—it simply failed. |
| devstral-24b | data-json-flatten-top-spender | 7 | thrash | lucky_fail | The agent never wrote the result to /workspace/result.txt; it only read the file and then stated 'Eve' in conversation, scoring 0. It failed to complete the required action despite multiple Python att |
| devstral-24b | data-json-flatten-top-spender | 8 | thrash | lucky_fail | The agent repeatedly failed to execute its Python code due to formatting issues (tabs/newlines) and never wrote the result to /workspace/result.txt, resulting in a score of 0. It guessed 'Eve' in its  |
| qwen3-coder-30b | data-json-flatten-top-spender | 1 | suspicious_pass | genuine | The agent read the JSON data, computed each user's total purchase amount correctly via python_eval, identified Dave as the top spender (555.25), and wrote the result to file using Python's file I/O wi |
| devstral-24b | data-log-parse-p95-latency | 1 | thrash | lucky_fail | The agent failed to write the result to /workspace/result.txt and its final stated answer (500ms) appears to be a guess from truncated data; it scored 0, so it didn't luck out on partial credit either |
| devstral-24b | multi-config-driven-transform | 1 | thrash | lucky_fail | The agent failed (score 0), never wrote the output file via fs_write as required, and hardcoded config values instead of reading them dynamically. It thrashed through 12 repeated python_eval calls wit |
| devstral-24b | multi-config-driven-transform | 3 | thrash | lucky_fail | The agent failed the task (score 0), never wrote the output file via fs_write, and thrashed through 12 turns repeating nearly identical hardcoded python_eval calls without making progress. |
| devstral-24b | multi-config-driven-transform | 5 | thrash | lucky_fail | The agent repeatedly computed the result in python_eval across 10 turns but never called fs_write to write the output file as required, resulting in a score of 0. It failed to complete the task. |
