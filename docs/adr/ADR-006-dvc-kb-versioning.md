---
doc_id: adr-006-dvc-kb-versioning
title: 'ADR-006: DVC for KB versioning, MinIO remote'
zone: lab
kind: adr
status: active
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
tags:
- lab
- adr
- rag
- dvc
- superseded-partial
---
# ADR-006: DVC for KB versioning, MinIO remote

> **NOTE (2026-05-27):** The per-KB directory DVC pattern documented here is
> partially superseded by **ADR-007 (content-addressed RAG registry)**. New
> KBs route through the registry-bundle DVC pattern instead. This ADR remains
> the record of the prior state and the migration reference.

Status: accepted (partially superseded by ADR-007)
Date: 2026-05-27
Deciders: Matt Rachlin

## Context

By Phase 15.4 we have a single sealed knowledge base (`~/db/kb/bash/`,
4620 chunks, ~140 MiB on disk) used by `lab.rag` for retrieval. More are
coming. Two problems show up as soon as we want a second KB:

1. **No cross-machine reproducibility.** The KB lives outside git on
   `/home/m/db/`. If I rebuild the lab on another box, I'd have to re-crawl
   500+ sources and re-embed 1.74M tokens for every KB just to bring up a
   working retriever. Embedding alone is ~30 GPU-minutes per KB.
2. **No history.** Re-running the build pipeline silently overwrites the
   index. We can't roll back to a known-good revision when a rebuild
   regresses the eval.

The KB has three distinct content categories:

| Category | Size | Refetchable? |
|---|---|---|
| Sources (raw HTML/Markdown/PDF) | ~11 MB | yes — URLs + sha256 in manifest |
| Sources normalized (`*.md`) | ~7 MB | yes — derived from raw |
| Chunks (jsonl + embeddings) | ~44 MB | no — embeddings cost GPU time |
| LanceDB index | ~94 MB | no — built from chunks |
| Manifest, logs | <1 MB | n/a (transient) |

The first two are inputs; the last two are expensive derived artifacts.
We want versioning where it pays for itself.

We already operate a MinIO instance (`lab-minio`) for run traces. Adding a
new bucket to it costs nothing. We're a solo lab — no need for a SaaS DVC
remote.

## Decision

**Adopt DVC for KB versioning, with the local MinIO instance as the remote.**

Concretely:

- **Track:** the LanceDB `index/` directory and the `chunks/*.jsonl` files.
- **Don't track:** sources (raw or normalized), build logs, or agent traces.
  Sources are refetchable from URLs in the manifest; logs are transient.
- **Layout:** the canonical data lives in `kbs/<name>/{index,chunks}/` inside
  this git repo. `~/db/kb/<name>/{index,chunks}` are symlinks pointing back
  so existing consumers (`lab.rag`, integration tests) keep working.
- **Pointer files:** `kbs/<name>/index.dvc` and `kbs/<name>/chunks.dvc` are
  in git. `dvc pull` restores the data; `dvc push` uploads new revisions.
- **Remote:** `s3://lab-dvc` on `http://localhost:9000` (MinIO). Credentials
  in `.dvc/config.local` (gitignored). Endpoint URL in `.dvc/config`
  (committed).
- **Build identity:** `tools/bump_kb_version.py` stamps a sortable
  `YYYYMMDD-HHMMSS-xxxx` token into the manifest's `kb_version` field before
  each republish. The manifest itself is tracked normally in git (it's a
  small YAML file with the source URL list).

## Consequences

**Easier**

- `just kb-pull <name>` on a fresh checkout gives a working KB in seconds.
- `git log kbs/<name>/index.dvc` shows the history of every revision; the
  md5 inside each pointer file is the integrity check.
- New KBs follow the same pattern — no schema design needed per KB.
- MinIO is already running; no new infrastructure.

**Harder**

- Two-step publish: `bump_kb_version → dvc add → dvc push → git commit`.
  We wrap it in `just kb-publish <name>`, but the workflow is still longer
  than "just rebuild and forget".
- DVC's symlink rules forced the data into the repo (we moved
  `~/db/kb/<name>/{index,chunks}` into `kbs/<name>/` and symlinked back).
  DVC 3.x removed external outputs, so the "pointer files in lab repo,
  data in ~/db/kb" pattern from the original plan didn't work. The current
  setup is functionally equivalent.

**Risks**

- The MinIO secret is a single point of failure. We back the MinIO data dir
  up nightly (`just backup`), so a lost MinIO doesn't lose the KB.
- The DVC cache (`.dvc/cache/`) on the working machine duplicates the data.
  For one KB that's ~140 MiB extra; not a concern at current scale.
- The absolute symlinks (`~/db/kb/<name>/{index,chunks}` → `/data/lab/code/...`)
  break if either path moves. We accept this; the lab is on a single box.

## Considered alternatives

- **git-lfs.** Rejected. LFS works for blobs but is awkward for directories
  with many files, and we'd need a separate LFS server. DVC was designed
  for this case.
- **Plain rsync to MinIO.** Rejected. No content addressing, no atomic
  revisions, no `git log`-style history.
- **Track everything (including sources).** Rejected. ~11 MB of refetchable
  inputs isn't worth the friction. The manifest already records each URL
  + sha256; the pipeline is deterministic.
- **External SaaS DVC remote (S3, GDrive).** Rejected. MinIO is already
  running and backed up; no reason to add a vendor dependency for a
  single-box solo lab.
- **Init DVC inside `~/db/kb/`** (option A from the original plan). Rejected.
  That would make the KB its own git repo, separate from the lab — pull
  workflows become "clone two repos". Keeping pointer files in the lab
  repo means one `git pull` brings down both the code and the pointers.
- **DVC's external outputs** (option C from the original plan). Tried;
  DVC 3.x removed support for external outputs (`ERROR: Cached output(s)
  outside of DVC project`). The current move-and-symlink-back layout
  achieves the same UX (data still accessible from `~/db/kb/<name>/`)
  without fighting DVC.

## References

- [`kbs/README.md`](../../kbs/README.md) — operational workflow
- [`kbs/bash/README.md`](../../kbs/bash/README.md) — first KB under DVC
- [DVC docs — Importing external data](https://dvc.org/doc/user-guide/data-management/importing-external-data)
