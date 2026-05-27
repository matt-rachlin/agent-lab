# lab-sweep

Sweep + analysis:
- `lab.sweep.config` — Hydra config dataclasses
- `lab.sweep.preflight` — quota/budget gates
- `lab.sweep.runner` — drive a sweep through inspect_bridge
- `lab.analyze.queries`, `lab.analyze.stats`, `lab.analyze.report` — post-hoc analysis on DuckDB+Postgres

## Gotchas
- Sweep runner imports `lab.inspect_bridge` to launch tasks; this is the heaviest dep chain.
- analyze.queries uses DuckDB postgres_scanner — needs Postgres reachable.
