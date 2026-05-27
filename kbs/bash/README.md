---
doc_id: kbs-bash-readme
title: kbs/bash — GNU Bash KB
zone: lab
kind: readme
status: active
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
tags:
- lab
- rag
- dvc
- reference
---
# `kbs/bash/` — GNU Bash KB

Authoritative knowledge of the GNU Bash shell — syntax, builtins, parameter
expansion, redirection, pipelines, job control, conditional expressions,
arrays, scripting idioms, debugging. Intended for AI-agent retrieval.

## Stats (first DVC publish, 2026-05-27)

- 498 sources crawled, 4620 chunks, 1.74M embedded tokens
- Index: 5 files, ~93 MiB (LanceDB `chunks.lance/`)
- Chunks: 4 JSONL files, ~44 MiB (raw / enriched / embedded / enrichment cache)
- Status in manifest: `sealed`

## Authority order

1. GNU Bash manual (primary)
2. Bash manpage
3. Greg's Wiki / BashGuide
4. Advanced Bash-Scripting Guide
5. ShellCheck wiki (per-SC-code rationale)

## Refresh procedure

```bash
# 1. Rebuild source plan, fetch, chunk, embed (existing pipeline)
lab kb build bash      # or whichever entrypoint applies

# 2. Stamp the manifest with a new version token
uv run python tools/bump_kb_version.py kbs/bash/manifest.yaml

# 3. Publish to MinIO via DVC
just kb-publish bash

# 4. Commit
git add kbs/bash/index.dvc kbs/bash/chunks.dvc kbs/bash/.gitignore \
        kbs/bash/manifest.yaml ~/db/kb/bash/manifest.yaml
git commit -m "kb(bash): refresh $(date +%Y-%m-%d)"
```

## Cross-machine pull

```bash
just kb-pull bash
# verifies md5 of every file against the .dvc pointer
```

## Round-trip verified

On 2026-05-27, a fresh `git clone` + `dvc pull kbs/bash/{index,chunks}.dvc`
restored 5 + 4 files matching the originals byte-for-byte (verified via
`diff -rq` and md5 of `chunks.jsonl`/`chunks.embedded.jsonl`).

## See also

- [`kbs/README.md`](../README.md) — overall DVC-for-KBs workflow
- [`manifest.yaml`](manifest.yaml) — sources, models, build budget, stats
- [ADR-006](../../docs/adr/ADR-006-dvc-kb-versioning.md) — design rationale
