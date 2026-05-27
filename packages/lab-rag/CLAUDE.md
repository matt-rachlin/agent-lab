# lab-rag

RAG stack:
- `lab.rag.chunker`, `lab.rag.embedder`, `lab.rag.index` — corpus build pipeline
- `lab.rag.fetchers.*` — HTML/PDF/SPA content fetchers
- `lab.rag.rerank`, `lab.rag.rerank_client`, `lab.rag.rerank_server` — Phase 7 host service
- `lab.rag.cache` — query cache
- `lab.rag.hype`, `lab.rag.expand` — query enrichment

## Gotchas
- The heavy reranker (sentence-transformers/torch) is host-only — sandbox cells
  talk to `rerank_server` over HTTP via `rerank_client`.
- LanceDB is on-disk; index path comes from `lab.core.settings`.
