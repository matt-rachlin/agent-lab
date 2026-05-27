---
doc_id: code-readme
title: lab
zone: lab
kind: readme
status: active
owner: m
created: '2026-05-25'
last_updated: '2026-05-25'
last_verified: '2026-05-25'
tags:
- lab
- readme
---
# lab

Solo AI/ML research lab. Comparing agentic workflows across models, configurations, and prompts under hardware constraints (RTX 3080 Ti, 12 GB VRAM).

See [`RESEARCH_OPS_PLAN.md`](/home/m/.local/portfolio-staging/RESEARCH_OPS_PLAN.md) for the master plan and the [`docs/`](./docs/) tree for the lab notebook (daily logs, experiments, findings, ADRs).

## Quick start

```bash
just bootstrap      # install dependencies
just db-init        # create Postgres lab DB + run migrations
just services-up    # MinIO + MLflow + LiteLLM proxy containers
just check          # ruff + mypy + tests
```

## Status

Phase 0 (foundations) in progress.

## License

MIT.
