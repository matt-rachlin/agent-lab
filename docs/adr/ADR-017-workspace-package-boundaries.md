---
doc_id: adr-017-workspace-package-boundaries
title: 'ADR-017: Workspace partition and package boundaries'
zone: lab
kind: adr
status: draft
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags: [lab, adr, architecture, packages, workspace]
---
# ADR-017: Workspace partition and package boundaries

Status: proposed
Date: 2026-06-14
Deciders: Matt Rachlin

## Context

Wave-1 architecture review (2026-06-14) identified that the 8-package uv workspace
has grown organically and lacks a written dependency contract. The proposed
`lab-platform` split in ADR-016 §6 adds a ninth member but has no canonical home
in the dependency order. Without a written boundary spec, reviewers cannot tell
whether a new import is a clean dependency or a cycle, and new packages land
without a principled test for whether they belong.

Current workspace members: `lab-core`, `lab-rag`, `lab-agent`, `lab-eval`,
`lab-inspect`, `lab-sweep`, `lab-observability`, `lab-cli`.

## Decision

### Package dependency order (strict DAG, no cycles)

```
lab-core
  └── lab-rag
  └── lab-agent
  └── lab-observability
        └── lab-eval
              └── lab-inspect
              └── lab-sweep
                    └── lab-cli
```

`lab-core` has zero internal deps. Every other package may import from packages
strictly below it in this order. No upward imports are permitted.

`lab-platform` (ADR-016 §6, future) slots between `lab-agent` and `lab-eval`:
it may import `lab-core`, `lab-rag`, `lab-agent`; it may not import `lab-eval`,
`lab-inspect`, `lab-sweep`, or `lab-cli`.

### Namespace contract

All public symbols live under `lab.<package>.<module>` where `<package>` matches
the uv member name minus the `lab-` prefix (e.g. `lab.core`, `lab.rag`,
`lab.agent`). Private helpers go under `lab.<package>._<module>`. Cross-package
imports MUST use the public namespace; never import from `_`-prefixed modules
across package boundaries.

### Test for adding or merging packages

Before adding a new package: confirm (a) it has a single coherent responsibility
that doesn't fit any existing member; (b) its dependency set is a strict subset
of existing members below its intended slot; (c) it introduces no new cycle.
Before merging packages: confirm the merged result stays under 500 LOC per module
and the combined responsibility is still coherent.

### Enforcement

`pre-commit` runs `uv run python -c "import lab.<pkg>"` for each member. A failing
import that previously passed is a CI break. `ruff` is configured to flag
`lab.<higher>` imports from `lab.<lower>` via `banned-module-level-imports`
in `pyproject.toml` (future: once the DAG above is ratified and encoded).

## Consequences

- Easier: cross-package changes are self-documenting; ADR-016's `lab-platform`
  slot is unambiguous; cycle detection becomes mechanical.
- Harder: enforcing the DAG order in ruff requires maintaining a per-package
  `banned-module-level-imports` list; must be updated when packages are added.
- Risks: existing code may already have soft violations that only show at import
  time, not at lint time. An audit pass is needed before CI enforcement.

## Considered alternatives

- **No explicit DAG** — current state; accumulates invisible cycles.
- **Single monolith `lab` package** — simpler but collapses eval, sweep, and core
  concerns into one package, which ADR-016 explicitly rejects for the platform split.
- **src-layout flat namespace** — investigated; the `lab.<pkg>` two-level namespace
  is already in use and matches the `uv workspace` member names cleanly.
