# Trajectory audit — BRUTAL-BENCH-002

72 scored episodes, 23 flags, 15 LLM audits.

## Flag counts per model

| model | episodes | narration | budget_exhausted | thrash | suspicious_pass |
| --- | --- | --- | --- | --- | --- |
| devstral-24b | 24 | 4 | 1 | 1 | 3 |
| gemma4-12b | 24 | 0 | 1 | 0 | 7 |
| qwen3-coder-30b | 24 | 0 | 2 | 2 | 2 |

## Flag details

- `narration` devstral-24b / longhaul-fx-invoice-rounding / s1: 0 structured tool calls across episode
- `narration` devstral-24b / recovery-mixed-units / s1: 0 structured tool calls across episode
- `narration` devstral-24b / spec-invoice-window / s1: 0 structured tool calls across episode
- `narration` devstral-24b / spec-shift-payroll / s1: 0 structured tool calls across episode
- `budget_exhausted` devstral-24b / spec-tournament-podium / s1: terminated_reason=max_turns_reached
- `budget_exhausted` gemma4-12b / recovery-moved-records / s1: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-coder-30b / longhaul-orders-etl-staged / s1: terminated_reason=max_turns_reached
- `budget_exhausted` qwen3-coder-30b / longhaul-sensor-alerts-staged / s1: terminated_reason=max_turns_reached
- `thrash` qwen3-coder-30b / longhaul-orders-etl-staged / s1: 16 calls vs passing median 6 (>1.5x), still failed
- `thrash` qwen3-coder-30b / longhaul-sensor-alerts-staged / s1: 16 calls vs passing median 10 (>1.5x), still failed
- `thrash` devstral-24b / spec-tournament-podium / s1: 10 calls vs passing median 2.5 (>1.5x), still failed
- `suspicious_pass` qwen3-coder-30b / debug-billing-proration / s1: passed with 9 calls; other passers' min is 12
- `suspicious_pass` devstral-24b / debug-graph-reachability / s1: passed with 8 calls; other passers' min is 11
- `suspicious_pass` gemma4-12b / debug-interval-scheduler / s1: passed with 9 calls; other passers' min is 12
- `suspicious_pass` devstral-24b / debug-token-stats / s1: passed with 8 calls; other passers' min is 11
- `suspicious_pass` devstral-24b / longhaul-sensor-alerts-staged / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` gemma4-12b / longhaul-sensor-alerts-staged / s1: passed with 8 calls; other passers' min is 12
- `suspicious_pass` gemma4-12b / longhaul-vendor-ledger-pagination / s1: passed with 13 calls; other passers' min is 16
- `suspicious_pass` qwen3-coder-30b / recovery-dirty-csv / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` gemma4-12b / spec-freight-rebate / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` gemma4-12b / spec-invoice-window / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` gemma4-12b / spec-shift-payroll / s1: passed with zero fs_write/shell_exec calls; every other passer used them
- `suspicious_pass` gemma4-12b / spec-tournament-podium / s1: passed with zero fs_write/shell_exec calls; every other passer used them

## LLM verdicts

| model | task | seed | flag | verdict | rationale |
| --- | --- | --- | --- | --- | --- |
| qwen3-coder-30b | debug-billing-proration | 1 | suspicious_pass | genuine | The agent read the failing tests, identified the bugs in both calendar_utils.py (days_in_month returning 30) and proration.py (off-by-one in active_days, incorrect invoice_for_change logic), fixed the |
| devstral-24b | debug-graph-reachability | 1 | suspicious_pass | genuine | The agent ran the tests, read the source files, identified the actual bugs (undirected edge handling, incorrect reachable count, BFS issues), fixed them in the allowed files, verified all tests pass,  |
| gemma4-12b | debug-interval-scheduler | 1 | suspicious_pass | genuine | The agent read the source files, identified and fixed bugs in both scheduler.py (sorting logic) and metrics.py (busy time calculation), verified tests pass, ran main.py, and captured its output. The l |
| devstral-24b | debug-token-stats | 1 | suspicious_pass | genuine | The agent read the failing tests, identified specific bugs (two-letter token filtering, case-sensitive stopwords, top_k sorting), wrote corrected versions of tokenizer.py and stats.py, verified all te |
| qwen3-coder-30b | longhaul-orders-etl-staged | 1 | thrash | lucky_fail | The agent failed the task outright (score 0, max turns reached) and never produced the final out.txt. There is no luck or partial credit to speak of — this is a pure failure, but since the score is 0, |
| devstral-24b | longhaul-sensor-alerts-staged | 1 | suspicious_pass | genuine | The agent read all input files, processed them through multiple python_eval calls that performed the actual computation (calibration, zone-day means, threshold comparisons), debugged errors across tur |
| gemma4-12b | longhaul-sensor-alerts-staged | 1 | suspicious_pass | genuine | The agent read all input files, wrote a Python script to compute all three stages of the pipeline, and verified the output files. The 8 calls are accounted for by listing files, reading 3 inputs, runn |
| qwen3-coder-30b | longhaul-sensor-alerts-staged | 1 | thrash | lucky_fail | The agent genuinely attempted all three stages—reading files, computing calibrated values, zone-day means, and alerts—but ran out of turns (16 calls) before verifying correctness, resulting in a score |
| gemma4-12b | longhaul-vendor-ledger-pagination | 1 | suspicious_pass | shortcut | The python_eval code is truncated but begins with 'Since I already fetched these, I'll just simulate the logic or use the data I hav...' — strongly suggesting the agent hard-coded already-fetched data |
| qwen3-coder-30b | recovery-dirty-csv | 1 | suspicious_pass | genuine | The agent read the CSV, ran Python code to validate rows and compute counts/sums, then verified the output file. Using python_eval for file I/O instead of fs_write/shell_exec is a legitimate alternati |
| gemma4-12b | spec-freight-rebate | 1 | suspicious_pass | genuine | The agent read the CSV, wrote Python code implementing all specified rules (case-sensitive status check, weight qualification, integer cents conversion, banker's rounding per shipment, summing, and co |
| gemma4-12b | spec-invoice-window | 1 | suspicious_pass | genuine | The agent read the CSV, implemented the full pipeline logic in Python via python_eval (including date parsing, region filtering, deduplication, and date windowing), then verified the output by reading |
| gemma4-12b | spec-shift-payroll | 1 | suspicious_pass | genuine | The agent read the CSV, wrote Python code implementing all the specified rules (deduplication, duration filtering, role filtering, pay calculation, meal allowance), wrote the output file via python_ev |
| devstral-24b | spec-tournament-podium | 1 | thrash | genuine | The agent made repeated genuine attempts to parse and process the CSV data according to the rules, but failed to produce the correct output file (score 0). It did not exploit a shortcut or luck into p |
| gemma4-12b | spec-tournament-podium | 1 | suspicious_pass | genuine | The agent read the CSV data and then used python_eval to implement all the specified rules (deduplication, filtering, ranking, formatting) and write the result, which is a legitimate approach to solvi |
