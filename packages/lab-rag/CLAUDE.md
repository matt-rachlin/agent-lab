# lab-rag

RAG stack:
- `lab.rag.chunker`, `lab.rag.embedder`, `lab.rag.index` — corpus build pipeline
- `lab.rag.fetchers.*` — HTML/PDF/SPA content fetchers
- `lab.rag.rerank`, `lab.rag.rerank_client`, `lab.rag.rerank_server` — Phase 7 host service
- `lab.rag.cache` — query cache
- `lab.rag.hype`, `lab.rag.expand` — query enrichment
- `lab.rag.registry` — **(planned, Tier H2.5)** content-addressed RAG
  registry: `source_docs`, `chunks`, `embeddings`, `knowledge_bases`,
  `kb_chunks`, enrichment tables. Spec at
  `~/docs/specs/2026-05-27-rag-registry-spec.md`; decision at
  `docs/adr/ADR-007-rag-registry.md`. Once landed, KBs are views over
  the registry; the existing per-KB pipeline is delegated through it.

## Gotchas
- The heavy reranker (sentence-transformers/torch) is host-only — sandbox cells
  talk to `rerank_server` over HTTP via `rerank_client`.
- LanceDB is on-disk; index path comes from `lab.core.settings`.
- **(Coming with H2.5)** Once `lab.rag.registry` lands, prefer
  `lab.rag.registry.ingest(...)` over directly calling `chunker` + `embedder`
  + `index` — the registry handles dedup, embedding cache, and KB
  membership atomically. Direct chunker/embedder/index usage is reserved
  for migration tooling and one-off experiments.
