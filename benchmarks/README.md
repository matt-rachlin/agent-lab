---
doc_id: lab-benchmarks-readme
title: '`benchmarks/` — performance regression tracking'
zone: lab
kind: readme
status: active
owner: m
created: '2026-05-26'
last_updated: '2026-05-26'
last_verified: '2026-05-26'
tags:
- lab
- readme
- benchmarks
---
# `benchmarks/` — performance regression tracking

Phase 13.3. Micro-benchmarks for hot code paths. The goal is *not*
publishable timings — it's catching unintended performance regressions
between commits (e.g. someone disables a cache, or pulls in a slow
library).

## Bench cases

| File                       | Measures                                              | Skip condition                              |
|----------------------------|-------------------------------------------------------|---------------------------------------------|
| `bench_kb_query.py`        | `lab kb query bash "redirect stderr"` p50/p95 (n=20)  | GPU lease (`lab:gpu:lease`) non-empty       |
| `bench_sweep_cell.py`      | single-cell wall-time on `llama3.1-8b-q4` (n=3)       | Ollama at `localhost:11434` unreachable     |
| `bench_rerank.py`          | `LabReranker.rerank` p50/p95 on a 50-cand fixture     | Rerank service `127.0.0.1:8401/healthz` 4xx |

Each bench module exposes:

```python
def run() -> dict[str, float]:
    ...
```

Skipping is signalled by `raise BenchmarkSkipped("reason")`.

## Running

```bash
just bench           # all benches, append to history.csv
just bench-quick     # only KB query (no GPU, no Ollama)
python -m benchmarks.runner            # direct invocation
python -m benchmarks.runner --quick    # same as bench-quick
```

The runner:

1. Imports each `bench_*.py` and calls `run()`
2. Appends one CSV row per metric to `benchmarks/history.csv`:
   `timestamp,bench_name,metric,value,commit_sha`
3. Writes a human-readable `benchmarks/latest.md` snapshot
4. Computes the 7-day rolling median per `(bench_name, metric)` and
   compares to the latest value

## Regression thresholds

| Ratio (latest / 7d-median) | Action                                 |
|----------------------------|----------------------------------------|
| ≤ 1.20                     | OK (printed in green)                  |
| 1.20 – 1.50                | warning (printed in yellow, exit 0)    |
| > 1.50                     | failure (printed in red, exit 1)       |

The 7-day median is computed across all runs of that
`(bench_name, metric)` pair where `commit_sha` differs from the latest
(so re-running on the same commit doesn't drown out history).
The median is undefined (and the check is skipped) when fewer than
three historical samples exist.

## When regressions trigger investigation

A 1.20× warning is usually noise from GPU thermal state, background
load, or cache warmth. Two consecutive warnings on the same metric
without an obvious cause = investigate.

A 1.50× failure on KB query or rerank almost always points to:

- a cache that stopped hitting (check `RagCache` and rerank tier-2)
- a library update with a regression (check `uv.lock` diff)
- a serialization change (JSON vs msgpack, etc.)

## History format

`history.csv` is append-only. Columns:

| column      | type   | notes                                          |
|-------------|--------|------------------------------------------------|
| timestamp   | str    | ISO 8601 UTC, e.g. `2026-05-26T22:30:00Z`      |
| bench_name  | str    | e.g. `kb_query`                                |
| metric      | str    | e.g. `p50_sec`, `p95_sec`                      |
| value       | float  | metric value (seconds, ratio, count, etc.)    |
| commit_sha  | str    | `git rev-parse --short HEAD`; `unknown` if no git |

Don't rewrite history (regressions need historical context). To
re-baseline a metric, just keep writing new rows; the rolling median
shifts on its own after ~7 days.

## Adding a new bench

1. Create `benchmarks/bench_<name>.py` with a `run() -> dict[str, float]`
2. Use `raise BenchmarkSkipped(...)` for unavailable preconditions
3. Add a test for it under `tests/unit/test_benchmarks_*.py`
4. Document it in the table above
