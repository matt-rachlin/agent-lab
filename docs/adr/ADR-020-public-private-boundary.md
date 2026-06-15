---
doc_id: adr-020-public-private-boundary
title: 'ADR-020: Public/private repository boundary'
zone: lab
kind: adr
status: draft
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags: [lab, adr, security, public, private, gitignore, ci]
---
# ADR-020: Public/private repository boundary

Status: proposed
Date: 2026-06-14
Deciders: Matt Rachlin

## Context

`code/` (this repo, `github.com/matt-rachlin/agent-lab`) was made public
2026-06-10 for the local coding-agent benchmark campaign writeup. The repo now
has a dual life: research artifact (public) and operational infrastructure
(private concern). Wave-2 public/private review (2026-06-14) produced a full
inventory of what must never appear in the public repo, what is safe, and what
is conditional on scrubbing.

No written policy existed before this ADR. The current `.gitignore` blocks `.env`
and `*.db` but has no explicit rule for secrets, personal data, host-specific
paths, or cross-zone references.

## Decision

### Always public

- All source code under `packages/` that contains no credentials, host paths, or
  personal data.
- ADRs, experiment docs, finding docs, daily logs (after personal detail review).
- Sweep configs under `conf/sweep/` (model names and hyperparams are research
  artifacts, not secrets).
- `pyproject.toml`, `justfile`, `uv.lock`, `.github/workflows/ci.yml`.
- Model cards and dataset datasheets under `docs/model-cards/` and `docs/datasets/`.

### Always private (must never be committed)

- Credentials and keys: `LITELLM_MASTER_KEY`, MinIO secret, Ollama Cloud signin
  tokens, any `*.pem` / `*.key`.
- Personal data: email addresses, home directory paths that identify the user,
  system usernames in non-example contexts.
- Host-specific runtime state: `/data/lab/services/minio-secret`,
  `/data/lab/models/awq/` MANIFEST.json weight SHA256s (contain locally-quantized
  model fingerprints — public only after intentional publication).
- `conf/litellm-config.yaml` and `conf/llama-swap.yaml` if they contain any
  expanded credentials (they currently use `os.environ/` references — safe as-is,
  but must not be replaced with inline values).
- `.env`, `.env.local`, any `*secret*` files.

### Conditional (public after review/scrub)

- Daily logs: personal names, meeting notes, contractor details → redact before
  committing. Research findings and timings are public.
- Scripts that reference `/home/m/` or `/data/lab/` hard-coded paths → replace
  with env-var references or document as "host-specific" before committing.
- `sandbox-image.sha` in `conf/` — contains a Podman image digest. Currently
  committed and treated as public; update policy only if the image becomes private.

### `.gitignore` rules

The following entries are required (add to `.gitignore` if missing):

```
# Secrets and host-specific runtime state
.env
.env.local
*.key
*.pem
*secret*
conf/litellm-config.yaml.bak-*
conf/llama-swap.yaml.bak-*
/data/
services/minio-secret
```

Backup files (`*.bak-*`) in `conf/` are already present locally and must stay
out of the public history. The un-suffixed canonical configs are public (they use
`os.environ/` refs).

### CI gates

1. **gitleaks** — add `gitleaks/gitleaks-action` to `.github/workflows/ci.yml`
   as a required check on every PR. Baseline the current state once before
   enabling to avoid false positives from historical commits.
2. **Placeholder-SHA grep** — add a CI step that fails if any committed file
   contains the literal string `os.environ/` expanded to a real value (i.e., the
   pattern `[A-Za-z0-9_]+=sk-` or `password=<non-placeholder>`). This is a
   belt-and-suspenders guard against accidental credential expansion.
3. **Cross-zone reference ban** — add a CI grep step that fails if any file under
   `packages/` or `scripts/` contains `/home/m/`, `/data/lab/services/`, or
   `mattrachlin@` outside of `# example` or `# host-specific` comment blocks.

### Cross-zone reference policy

Code in this repo must not contain hard-coded references to sibling zones
(`~/workspaces/medivh/`, `~/workspaces/zur/`, `~/code/`). Use env vars
(`LAB_HOME`, `LAB_KB_ROOT`, etc.) defined in `lab.core.settings` for any
host-specific path. ADRs and docs may reference zone paths in prose (they are
not executable artifacts).

## Consequences

- Easier: a clear checklist for contributors; CI blocks accidental leaks before
  they reach GitHub.
- Harder: gitleaks baseline is one-time setup work; cross-zone grep will need
  exception comments in a few existing scripts that reference host paths for
  documentation purposes.
- Risks: the conditional category requires per-commit reviewer judgment. The only
  automated guard is gitleaks + the placeholder grep; personal data in prose
  (daily logs) is not machine-detectable.

## Considered alternatives

- **Keep repo private** — rejected: the public benchmark campaign writeup requires
  a public repo for credibility and reproducibility.
- **Split into public mirror + private operational repo** — considered; adds
  significant sync overhead for a solo researcher. The `os.environ/` reference
  pattern in config files is sufficient separation for current scale.
