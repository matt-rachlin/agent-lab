---
doc_id: kbs-overview
title: kbs/ — DVC-tracked knowledge bases
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
# `kbs/` — DVC-tracked knowledge bases

Knowledge bases (KBs) feed `lab.rag` retrieval. Each KB has heavy artifacts
(LanceDB index, chunk JSONL) that don't belong in git. We track them with
[DVC](https://dvc.org/), with the canonical data inside this repo and the
DVC remote on the local MinIO instance.

## Layout per KB

```
kbs/<name>/
├── index/            # NOT git-tracked; DVC-tracked (LanceDB)
├── index.dvc         # git-tracked DVC pointer
├── chunks/           # NOT git-tracked; DVC-tracked (JSONL)
├── chunks.dvc        # git-tracked DVC pointer
├── .gitignore        # git-tracked; ignores index/ + chunks/
├── manifest.yaml     # convenience symlink → ~/db/kb/<name>/manifest.yaml
├── README.md         # human notes for this KB (git-tracked)
└── CLAUDE.md         # agent-facing notes (git-tracked)
```

`~/db/kb/<name>/index` and `~/db/kb/<name>/chunks` are symlinks pointing
back into this repo so existing consumers (`lab.rag`, integration tests)
keep working unchanged.

## What we track vs. don't

| Tracked via DVC | Not tracked |
|---|---|
| `kbs/<name>/index/` (LanceDB index, ~90 MiB+) | `~/db/kb/<name>/sources/` (refetchable from URLs in the manifest) |
| `kbs/<name>/chunks/*.jsonl` (chunked + embedded text) | `~/db/kb/<name>/build.log`, `agent-trace/` (transient) |
| `kbs/<name>/index.dvc` + `chunks.dvc` (pointer files, in git) | `~/db/kb/<name>/sources/normalized/*.md` (derived) |

We deliberately do not version the raw sources — the manifest records each
source URL + sha256, and rebuilding the KB regenerates the same chunks.

## Adding a new KB

1. Copy `_template/` to `kbs/<new-name>/` and fill in `README.md` + `CLAUDE.md`.
2. Build the KB the usual way (`lab kb build <name>` or equivalent), producing
   `~/db/kb/<new-name>/index/` and `~/db/kb/<new-name>/chunks/`.
3. Move the data into the repo and symlink it back:
   ```bash
   mv ~/db/kb/<name>/index   kbs/<name>/index
   mv ~/db/kb/<name>/chunks  kbs/<name>/chunks
   ln -s /data/lab/code/kbs/<name>/index   ~/db/kb/<name>/index
   ln -s /data/lab/code/kbs/<name>/chunks  ~/db/kb/<name>/chunks
   ln -s ../../../../../home/m/db/kb/<name>/manifest.yaml \
         kbs/<name>/manifest.yaml
   ```
4. Stamp the manifest with a fresh build token:
   `uv run python tools/bump_kb_version.py kbs/<name>/manifest.yaml`
5. Publish: `just kb-publish <name>` (runs `dvc add` + `dvc push`).
6. Commit the pointer files: `git add kbs/<name>/{index,chunks}.dvc kbs/<name>/.gitignore`.

## Refreshing an existing KB

After rebuilding:

```bash
uv run python tools/bump_kb_version.py kbs/<name>/manifest.yaml
just kb-publish <name>
git commit -m "kb(<name>): refresh ($(date +%Y-%m-%d))"
```

`dvc add` re-hashes; only changed files are uploaded.

## Cross-machine reproducibility

```bash
git clone <repo> /path/to/lab-code
cd /path/to/lab-code
cp /elsewhere/.dvc/config.local .dvc/config.local   # MinIO creds
just kb-pull <name>
```

After `dvc pull`, the index and chunks files match byte-for-byte (md5
verified by DVC).

## See also

- [ADR-006](../docs/adr/ADR-006-dvc-kb-versioning.md) — design rationale
- [`justfile`](../justfile) — `kb-publish`, `kb-pull`, `kb-list`, `kb-status`
- [`_template/`](_template/) — starter files for a new KB
