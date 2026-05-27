---
doc_id: adr-007-rag-registry
title: 'ADR-007: Content-addressed RAG registry (KB-as-view)'
zone: lab
kind: adr
status: draft
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
depends_on:
  - doc:adr-001-storage-stack
  - doc:adr-006-dvc-kb-versioning
  - doc:2026-05-27-rag-registry-spec
tags:
  - lab
  - adr
  - rag
  - kb
  - registry
  - tier-h
---

# ADR-007: Content-addressed RAG registry (KB-as-view)

Status: proposed
Date: 2026-05-27
Deciders: Matt Rachlin
Companion spec: `~/docs/specs/2026-05-27-rag-registry-spec.md`

## Context

After Phases 6h, 7-12, and the EXP-004 series, the lab has:
- One sealed KB (bash, ~4600 chunks) and a roadmap that adds at least 3
  more in the near term (kubectl, git, claude-code per the kb-builder
  proving plan; plus dataset-derived KBs from Tier H5 acquisitions).
- A per-KB pipeline (fetch → normalize → chunk → embed → enrich →
  LanceDB) at `~/code/experimental/kb-builder/` and `packages/lab-rag/`.
- DVC versioning per-KB directory (ADR-006).

Three forces are now pulling in the opposite direction from the per-KB
pattern:

1. **Compute waste.** Multiple KBs will draw on overlapping sources
   (e.g., a Python KB and a general-developer KB both reference common
   tooling docs). Per-KB pipelines re-fetch, re-chunk, and re-embed the
   same bytes. At our current embedding cost of ~30 GPU-minutes per
   bash-scale KB, this compounds quickly.

2. **Experiment friction.** EXP-003c (embedding-model ablation) and
   EXP-004d (rerank on noisier KB) need clean A/B over the *same chunks*
   with different models or different rerank configurations. The per-KB
   pipeline makes this expensive — every model swap is a rebuild.

3. **Provenance and cross-KB analytics.** "Which sources show up in the
   most KBs?" "Which chunks are dead weight?" "What's the dedup ratio
   across our content?" — none answerable without a centralized,
   content-addressed view.

## Decision

Adopt a **content-addressed RAG registry** as the source of truth for
all source documents, chunks, embeddings, and enrichments. Knowledge
bases become metadata-only **views** over the registry, defined by a
chunk-membership table plus per-KB metadata.

Architectural shape:
- Postgres holds the registry tables (source_docs, chunks, embeddings,
  chunkers, embedding_models, knowledge_bases, kb_chunks,
  enrichment_kinds, chunk_enrichments, enrichment_embeddings)
- LanceDB holds vector data, one table per embedding model
- MinIO holds raw bytes + normalized text blobs
- DVC versions the registry as a single artifact bundle (revises ADR-006)

Per-KB CLIs and the `kb` MCP surface stay the same; the implementation
delegates to `lab.rag.registry`.

Full schema, hashing conventions, license model, and acceptance
criteria are in the spec at
`~/docs/specs/2026-05-27-rag-registry-spec.md`.

## Consequences

### Positive
- Embed-once, reuse-everywhere; first 5 overlapping KBs expected to
  produce 1.5×-3× compute savings, compounding from there
- KB-as-view: new KBs over existing content are metadata-only operations
- Clean A/B for embedding models and chunkers on identical chunks
- Stable provenance: every chunk traces to source bytes, license,
  fetcher, timestamp
- Re-embedding is incremental: enumerate missing `(chunk, model)` pairs
- Cross-KB analytics become trivial SQL
- License-policy enforcement at attach time

### Negative
- Schema migration of the bash KB is one-shot work (see spec
  §Migration) and must be verified retrieval-eval-equivalent before
  cutover
- Storage grows linearly with `chunker × model` combinations; needs
  garbage collection discipline
- A single registry tier becomes a coupling point — bugs affect every
  KB; canary discipline is required
- LanceDB single-table-per-model design needs sharding strategy beyond
  ~10M chunks (deferred until measured)
- Existing ADR-006 DVC pattern needs revision (this ADR supersedes the
  per-KB versioning portion)

### Neutral
- LanceDB stays as the vector backend
- `lab.rag` surface (chunker, embedder, fetchers, rerank, cache, hype,
  expand) stays; the orchestration of these moves through the registry
- The `kb` MCP server surface stays identical to consumers

## Alternatives considered

1. **Status quo (per-KB pipelines)**. Lowest implementation cost; highest
   ongoing compute cost; blocks the experiments we want to run.

2. **LangChain `CacheBackedEmbeddings` only**. Caches at the embedding
   layer but leaves chunking + source duplication unsolved. Half-measure.

3. **External feature store (FEAST, Hopsworks)**. Right shape but
   over-engineered for our scale (10M-100M chunks predicted, not
   billions). Operational overhead unjustified.

4. **Per-source-domain physical sharding** (e.g., one Postgres + LanceDB
   per source domain). Higher dedup at the cost of cross-domain analytics
   and KB construction. Deferred until measured to be needed.

5. **Vector lakehouse (Iceberg + LanceDB)**. Promising 2024-26 trend
   but tooling immature for our scale. Revisit at the 10M+ chunk
   threshold.

## Implementation tracking

- Spec: `~/docs/specs/2026-05-27-rag-registry-spec.md` (this ADR codifies
  it)
- Slot: Tier H, sub-phase H2.5 (~5-8 dev-days)
- Bash KB migration is the acceptance gate; experiments in Tier C
  (EXP-003c, EXP-004d) benefit immediately afterward

## Revisions to prior ADRs

This ADR partially supersedes:
- **ADR-006** (DVC for KB versioning, MinIO remote) — the per-KB
  directory DVC pattern is replaced by registry-bundle DVC. ADR-006 is
  not retired; it documents the prior state and the transition is
  spec'd above.
