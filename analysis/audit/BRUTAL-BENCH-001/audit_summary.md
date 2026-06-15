# Trajectory audit — BRUTAL-BENCH-001

72 scored episodes, 16 flags, 8 LLM audits.

## Flag counts per model

| model | episodes | narration | budget_exhausted | thrash | suspicious_pass |
| --- | --- | --- | --- | --- | --- |
| devstral-24b | 24 | 4 | 1 | 1 | 0 |
| gemma4-12b | 24 | 0 | 0 | 0 | 2 |
| qwen3-coder-30b | 24 | 0 | 3 | 1 | 4 |

## Flag details

- `narration` devstral-24b / longhaul-fx-invoice-rounding / s1: 0 structured tool calls across episode
- `narration` devstral-24b / recovery-mixed-units / s1: 0 structured tool calls across episode
- `narration` devstral-24b / spec-invoice-window / s1: 0 structured tool calls across episode
- `narration` devstral-24b / spec-shift-payroll / s1: 0 structured tool calls across episode
- `budget_exhausted` devstral-24b / spec-tournament-podium / s1: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-coder-30b / longhaul-orders-etl-staged / s1: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-coder-30b / longhaul-sensor-alerts-staged / s1: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-coder-30b / longhaul-vendor-ledger-pagination / s1: terminated_reason=max_turns_reached
- `thrash` qwen3-coder-30b / longhaul-sensor-alerts-staged / s1: 16 calls vs passing median 9 (>1.5x), still failed
- `thrash` devstral-24b / spec-tournament-podium / s1: 10 calls vs passing median 2.5 (>1.5x), still failed
- `suspicious_pass` qwen3-coder-30b / recovery-buggy-script / s1: passed with 5 calls; other passers' min is 8
- `suspicious_pass` qwen3-coder-30b / recovery-dirty-csv / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` qwen3-coder-30b / recovery-mixed-units / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` gemma4-12b / spec-shift-payroll / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` qwen3-coder-30b / spec-stock-audit / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` gemma4-12b / spec-tournament-podium / s1: passed with zero fs_write/shell_exec calls; every other passer used them

## LLM verdicts

| model | task | seed | flag | verdict | rationale |
| --- | --- | --- | --- | --- | --- |
| qwen3-coder-30b | longhaul-sensor-alerts-staged | 1 | thrash | genuine | The agent legitimately read all input files, processed data through Python across all three stages, and wrote output files without shortcuts. It simply failed to produce correct results within the tur |
| qwen3-coder-30b | recovery-buggy-script | 1 | suspicious_pass | genuine | The agent read the CSV, ran the buggy script (which reported an incorrect total of 19), identified the invariant violation, and computed the correct counts directly from the data to produce a valid re |
| qwen3-coder-30b | recovery-dirty-csv | 1 | suspicious_pass | genuine | The agent read the CSV, used python_eval to process and validate rows per the rules, and wrote the output file (likely within the Python code, which is truncated). The use of python_eval instead of fs |
| qwen3-coder-30b | recovery-mixed-units | 1 | suspicious_pass | genuine | The agent read the CSV file, then used python_eval to correctly process it—converting cents rows to dollars and summing—producing the right answer (509.54, 4 cents rows, 4 dollars rows). Hardcoding th |
| gemma4-12b | spec-shift-payroll | 1 | suspicious_pass | genuine | The agent used python_eval to compute the payroll and write out.txt via Python's file I/O (visible in the truncated code showing proper CSV parsing and deduplication logic), then verified the output w |
| qwen3-coder-30b | spec-stock-audit | 1 | suspicious_pass | genuine | The agent read the CSV, wrote Python code in python_eval to implement all the specified rules (unit factors, reject rows, sellable filtering, overstocked logic, tie-breaking, and output formatting), a |
| devstral-24b | spec-tournament-podium | 1 | thrash | lucky_fail | The agent repeatedly ran python_eval calls trying to compute the answer but never wrote the result to out.txt, resulting in a score of 0 (FAIL). It failed to complete the task, not a genuine solve or  |
| gemma4-12b | spec-tournament-podium | 1 | suspicious_pass | genuine |  |
