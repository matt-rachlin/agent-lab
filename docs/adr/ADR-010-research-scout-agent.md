---
doc_id: adr-010-research-scout-agent
title: 'ADR-010: Research-scout agent — outward intelligence, cited recs, human triage'
zone: lab
kind: adr
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags: [lab, adr, research-agent, scout]
---
# ADR-010: Research-scout agent

Status: proposed
Date: 2026-06-14
Deciders: Matt Rachlin

## Context
Pivot of the research agent's first goal from inward experiment-running to an
outward **research scout**: scan the world (papers, models, repos, news) for work
relevant to what we build, recommend (models / architectures / software), and log
recs for human triage. Lower risk (reads web, writes recs — no GPU jobs, no code
deploys). The experiment-loop agent is deprioritised, not cancelled (complementary
— the scout finds what's worth experimenting on).

## Decision
- **On-demand** scout (invoked, not a daemon). Sources: arXiv/papers, HuggingFace
  models, GitHub, blogs/news/X.
- **Relevance filter = the curated lab-profile** ([scout/lab-profile.md](../scout/lab-profile.md))
  + auto-pulled doc titles/TL;DRs (ADRs/findings/SETUP). The `kb` is off-limits.
- **Trust discipline carries over (ADR-008 ethos):** every recommendation MUST
  cite a real, fetched source (URL) + a confidence tag. No uncited claims, no
  invented papers. A rec without a verifiable source is discarded.
- **Output = a recommendation queue** (cited, categorised, confidence-tagged,
  status=new) that a human triages. Recs are suggestions, never auto-actions.
- **Control (lighter Stage-0 #10):** least-privilege identity (no push/deploy/
  cloud-write), append-only rec/audit log, web+LLM call budget, kill switch.
- **Dedup:** never re-surface an already-logged item or a known finding.

## Consequences
- Easier: immediate value (real recs), low blast radius, reuses the trust ethos.
- Harder: relevance quality depends on profile upkeep; web claims need verification
  (cite+fetch); noise control (esp. blogs/X).
- Risks: hallucinated sources (mitigate: fetch+cite required), low-signal firehose
  (mitigate: lab-profile filter + dedup + confidence threshold).

## Considered alternatives
- Scheduled daemon — deferred (on-demand chosen; simpler, no autonomous runtime).
- Free-form summaries — rejected (structured, cited, categorised recs are triageable).

## Running a scan (the repeatable loop)
The `lab scout` command provides the durable scaffolding; a search-capable agent drives the scan:
1. `lab scout context` — emits grounding (charter + lab-profile + ADR/finding titles + the dedup list).
2. A search-capable agent (Claude Code, with web tools) reads that, scans the sources
   (arXiv / HuggingFace / GitHub / web), and for each relevant, **verifiable** finding runs
   `lab scout add <url> --title … --category … --why … --confidence …`
   (deduped on `source_url`; cite a real fetched source — no uncited claims, ADR-008 ethos).
3. Triage: `lab scout list [--status new]`; advance `status` (new→triaged→actioned/rejected) as processed.
On-demand by design; a `/scout` skill can wrap step 2.
