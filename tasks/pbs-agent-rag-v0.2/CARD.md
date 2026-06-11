---
zone: lab
doc_id: pbs-agent-rag-v0-2-card
title: pbs-agent-rag-v0.2 — task suite
kind: card
status: active
owner: m
created: 2026-06-11
last_updated: 2026-06-11
---

## Purpose

<!-- BEGIN HAND -->
Agentic-RAG / deep-research slice: 14 multi-hop questions over the sealed
bash KB whose answers combine facts from 2-3 DISTINCT chunks (8 two-hop,
4 three-hop, 2 adversarial where the naive query surfaces a
plausible-but-wrong near-miss first). The agent must write
`/workspace/out.txt` as `answer=<exact answer>` plus one `cite=<chunk_id>`
line per supporting chunk. Predicates check the answer line and the key
citation(s); deeper citation validity (existence in the KB, semantic
support, fabrication rate) is scored post-hoc by
`scripts/citation_check.py`. Ground truth (hop regexes, gold chunk sets)
lives in `ground_truth.json`; `verify_rag.py` in this directory re-checks
the corpus-level multi-hop properties (no single chunk covers all hops;
a single top-8 kb_query leaves coverage incomplete for 12/14 tasks).
<!-- END HAND -->

<!-- BEGIN AUTOGEN -->

## Categories

- `rag` — 14 task(s)

## Difficulty distribution

- hard: 6
- medium: 8

## Tools used (union across tasks)

- `fs_write`
- `kb_query`

## Pre-reg shape

- success_predicate types:
  - `all_of` — 14 task(s)
- rubric types:
  - `tool_call` — 14 task(s)

## Experiments using this suite

None on record (no `experiment_runs` rows reference this suite).

## Findings citing this suite

No findings yet.

<!-- END AUTOGEN -->

## Known limitations

<!-- BEGIN HAND -->
- Chunk ids are tied to the 2026-05-26 sealed bash KB build; re-chunking
  the KB invalidates both the pinned predicate ids (POSIX trap table
  `01KSHSNCTPNG46M876KPZ81751`, wooledge job-control
  `01KSHSNCW876EV9G679ZTBEABF`) and `ground_truth.json`. Re-run
  `verify_rag.py` after any KB refresh.
- The bash corpus mirrors the GNU manual ~7x (html/texinfo/info/man), so
  most facts have several acceptable supporting chunks — predicates pin
  exact ids only where support is unique; everything else is delegated
  to `scripts/citation_check.py`.
- Four whole-document mega-chunks (390-620 KB; chunker artifact) match
  almost any regex; they are excluded from ground-truth support sets and
  scored as "weak" citations by citation_check.
- Strong models likely KNOW these bash facts from pretraining; the suite
  measures grounded citation behaviour, not raw knowledge — answers
  without valid cites fail the predicate, fabricated cites are caught by
  citation_check.
- 7/14 tasks lean on the signal/exit-status fact family (the cleanest
  genuinely multi-hop structure this corpus offers); treat per-family
  scores accordingly.
- rag2-star-join-default-ifs and rag2-pipestatus-first ARE coverable by
  one lucky top-8 query (the allowed 2-of-14 budget); the other 12 are
  not.
<!-- END HAND -->
