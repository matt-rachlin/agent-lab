---
doc_id: lab-code-claude
title: '`/data/lab/code/` — lab repo'
zone: lab
kind: claude
status: active
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
tags:
- lab
- claude
- workspace
---

# `/data/lab/code/` — lab implementation

Symlinked as `~/lab/code/`. This is the code repo for the lab; see
`~/lab/CLAUDE.md` for the lab zone identity. The session-start hook for
this directory is configured in `~/system/configs/claude/workspaces.json`.

## Workspace layout (Phase 15.1)

Source is split into 8 uv workspace member packages, all under the
`lab.*` PEP 420 namespace:

```
/data/lab/code/
├── pyproject.toml          # workspace root umbrella, declares [tool.uv.workspace]
├── packages/
│   ├── lab-core/           # lab.core.*     settings, manifest, notify, gpu_lease,
│   │                       #                llm, minio_io, daily_log + SQL migrations
│   ├── lab-rag/            # lab.rag.*      chunker, embedder, LanceDB index,
│   │                       #                fetchers, rerank (client/server), cache,
│   │                       #                hype, expand
│   ├── lab-agent/          # lab.agent.*    gVisor sandbox, ToolPool, tool servers
│   ├── lab-eval/           # lab.eval.*     evaluator framework, judge, builtins;
│   │                       # lab.tasks.*    task registry
│   ├── lab-inspect/        # lab.inspect_bridge.*  Inspect AI adapter + scorers
│   ├── lab-sweep/          # lab.sweep.*    Hydra sweep runner + preflight;
│   │                       # lab.analyze.*  DuckDB queries, stats, reports
│   ├── lab-observability/  # lab.observability.*   gpu_exporter, spend, quota
│   └── lab-cli/            # lab.cli        Typer CLI;
│                           # lab.experiment, lab.finding, lab.models.*
├── tests/                  # unit + integration tests (single tree, multi-package)
├── conf/                   # Hydra configs
├── benchmarks/             # perf regression benches
└── apps/eval-dashboard/    # Streamlit dashboard (Phase 15.3)
```

Dependency order (least-to-most): lab-core → lab-rag → lab-agent → lab-eval
→ lab-inspect → lab-sweep / lab-observability → lab-cli.

## Gates / verification

```
uv sync --all-extras
uv run ruff check packages/ tests/
uv run ruff format --check packages/ tests/
scripts/mypy-precommit.sh       # mypy --strict on packages/
uv run pyright packages         # 5 pre-existing errors (Phase 13 baseline)
uv run pytest tests/unit/ -q    # 513 tests as of Phase 15.1.9
just check                      # full lab gate
```

`scripts/mypy-precommit.sh` wraps `uv run mypy packages` (which mypy can't
do directly without `mypy_path` + `explicit_package_bases`, both set in
`mypy.ini`).

## Adding a new module

- Decide which workspace package owns it (its dependencies determine which).
- Drop the file under `packages/lab-<name>/src/lab/<name>/<module>.py`.
- Imports inside the package use `from lab.<name>.<module> import X`.
- Cross-package imports likewise use the canonical `lab.<name>.X` path.
- Don't add a `src/lab/__init__.py` in any package — PEP 420 namespace
  requires its absence.

## Phase 15.1 history

The workspace split happened across commits af78e74 .. 7f1a24e (May 27, 2026).
See those commit messages for the per-package move details. Two early
commits (15.1.1 skeleton + 15.1.3 lab-rag) had their staged contents
absorbed by sibling agent commits running in parallel — content is
correct, attribution noted in those commits.
