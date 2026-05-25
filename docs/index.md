# lab

Solo AI/ML research lab — comparing agentic workflows across models, configurations, and prompts under hardware constraints (RTX 3080 Ti, 12 GB VRAM).

**This site is local-only.** Browse via `lab docs serve` (default port 8001). Not published externally.

## Navigation

- **Log** — daily working notes. Open today's first: `lab today`.
- **Experiments** — pre-registered plans; authored before any sweep runs.
- **Findings** — durable distilled results, linked to the experiments that produced them.
- **ADRs** — Architecture Decision Records (Nygard format).
- **SOPs** — Standard Operating Procedures for repeatable workflows.
- **Protocols** — formal methods (judge calibration, reliability sweep, contamination check).
- **Post-mortems** — what happened on major experiments.
- **References** — citations + per-paper notes; `references.bib` for tooling.
- **Datasets** — datasheets per dataset (Gebru et al. 2018).
- **Models** — model cards per artifact we produce (Mitchell et al. 2018).

The left sidebar auto-updates as new files land in each section (powered by `mkdocs-awesome-pages-plugin`). Each page shows its last-modified date via `mkdocs-git-revision-date-localized-plugin`.

## Helpful CLI

```text
lab today                    # open today's daily log
lab exp list                 # all experiments + pre-registration status
lab finding list             # all findings, newest first
lab docs recent              # 10 most recently modified docs
lab docs serve [--port N]    # local-only mkdocs server
lab analyze report SLUG      # markdown report for an experiment
```

## Status

See [`BUILD_LOG.md`](file:///home/m/.local/portfolio-staging/BUILD_LOG.md) for current phase and recent activity, and [`RESEARCH_OPS_PLAN.md`](file:///home/m/.local/portfolio-staging/RESEARCH_OPS_PLAN.md) for the master plan.
