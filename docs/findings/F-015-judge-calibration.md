---
doc_id: f-015-judge-calibration
title: 'F-015: LLM judges vs machine-verified ground truth (n=240 episodes).
  Best judge 92.8% accurate; worst passes 1 in 5 verified failures; self-family
  leniency confirmed in qwen (FPR 0.250 same-family vs 0.128); judges degrade
  on SHORT episodes, not long ones; stated confidence carries almost no signal.'
zone: lab
kind: finding
status: active
owner: m
created: '2026-06-11'
last_updated: '2026-06-11'
last_verified: '2026-06-11'
depends_on:
- kind: code
  target: lab:scripts/judge_calibration.py
- kind: artifact
  target: lab:analysis/judge-calibration/SUMMARY.md
- kind: artifact
  target: lab:analysis/judge-calibration/confusion_matrices.csv
- kind: doc
  target: f-012-agentic-tool-calling-failure-modes
tags:
- lab
- finding
- findings
- judge-calibration
- meta-evaluation
- llm-judge
- confidence-high
- importance-8
---

# F-015: How accurate are LLM judges? Measured against owned ground truth

## TL;DR

The lab's machine-verified episodes make judge error directly
measurable. 240 episodes (151 pass / 89 fail, stratified from 364
across 4 experiments), 3 frontier-class cloud judges, full
trajectories, temperature 0:

| judge | accuracy | **FPR on failures** | FNR on passes |
|---|---|---|---|
| glm-5.1 | **0.928** | 0.081 | 0.066 |
| gpt-oss-120b | 0.879 | **0.056** | 0.159 |
| qwen3-coder-480b | 0.874 | **0.195** | 0.086 |

- **The dangerous error — passing a real failure — ranges 6–20%.**
  qwen3-coder-480b passes 1 in 5 machine-verified failures. A team
  using it as their eval judge would ship agents that fail twice as
  often as their dashboards claim.
- **Self-family leniency is real:** qwen judging qwen-family episodes
  shows FPR 0.250 vs 0.128 on other families; its judged pass rate
  (0.624) exceeds ground truth (0.550) only there.
- **Judges fail on SHORT episodes, not long ones** — the opposite of
  the folk "verbosity bias": every judge's worst tercile is the short
  one (early-terminating failures give too little evidence; judges
  guess pass). `code` tasks are the FPR hotspot for all three
  (0.21/0.17/0.31).
- **Stated confidence is near-useless:** qwen put 100% of verdicts at
  confidence ≥90 (87.4% accurate there). glm is best-calibrated and its
  rare sub-90 confidences genuinely flag trouble (80–89 bucket: 55.6%
  correct). Practical rule: treat any judge confidence < 90 as "do not
  trust this verdict."
- **Ensembling won't save you:** pairwise κ 0.70–0.84 — the judges
  share the short-episode/code blind spot, so majority vote inherits
  it.

## Method

scripts/judge_calibration.py: full conversation (tool results
truncated at 2 KB each, marked) + task statement, predicate and ground
truth withheld; JSON verdict {pass|fail, confidence, rationale}; one
strict retry on parse failure (5 abstains total / 720 calls); resumable
response cache. 721k prompt + 190k completion tokens via the local
litellm proxy's Ollama Cloud lanes.

## Caveats

- Ground truth = end-state predicates. F-012-style "shortcut passes"
  (trajectory_audit found one) mean a judge could be *right* to fail an
  episode the predicate passed — at most a handful of cells here, but
  it bounds "judge error" from below.
- All three judges came through one provider path (Ollama Cloud);
  provider-side serving differences are unobservable.
- Single prompt template for judging; judge prompt-sensitivity (cf.
  F-013) was not swept.

## Consequences

- Lab default judge: **glm-5.1**, and only where predicates can't
  reach; verdicts below confidence 90 are escalated to predicates or
  human review.
- Any lab use of judge scores must report the judge's measured FPR
  alongside (this study is the reference).
- Public writeup candidate: "We measured LLM judges against 240
  machine-verified agent runs" — few teams own the ground truth to do
  this.
trust_level: unverified
