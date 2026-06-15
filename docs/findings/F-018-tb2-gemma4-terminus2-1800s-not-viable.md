---
doc_id: f-018-tb2-gemma4-terminus2-1800s-not-viable
title: 'F-018: gemma4-12b + terminus-2 + 1800s/trial cap is not viable on Terminal-Bench 2.0 (17/17 AgentTimeoutError)'
zone: lab
kind: finding
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
depends_on:
- kind: doc
  target: log-2026-06-13
- kind: doc
  target: adr-015-sglang-serving
tags:
- lab
- finding
- terminal-bench
- tb2
- gemma4
- agentic
- timeout
- local-agent
---

# F-018: gemma4-12b + terminus-2 + 1800s/trial cap is not viable on Terminal-Bench 2.0

Date: 2026-06-14
Confidence: high (result is unambiguous — 100% timeout rate)
trust_level: unverified

## Claim

Running gemma4-12b via the terminus-2 scaffold on Terminal-Bench 2.0 with a
1800s (30-minute) per-trial agent cap yields **17/17 AgentTimeoutError** —
every completed trial exhausts its budget before producing a reward. This
specific combination is not viable for TB2 evaluation.

Note on scope: the broader claim "local 12B agents are not viable on TB2" is
**not supported** by this evidence. Three substantial levers remain untested
(see Open questions).

## Evidence

From the 2026-06-13 daily log (TB2-LOCAL-001 session finding):

- Scaffold: terminus-2 + gemma4-12b via litellm proxy
- Suite: Terminal-Bench 2.0, full suite, serial (-n 1), 1800s/trial cap
- Trials completed before kill: 17
- Outcome: 17/17 AgentTimeoutError
- Mechanism: TB2 tasks require many agent turns over long terminal-output
  contexts. A local 12B on the RTX 3080 Ti cannot finish a task inside
  30 minutes; every trial exhausts its wall-clock budget mid-LLM-call.
- Not a routing failure: single gemma4-12b proxy calls are fast; the
  bottleneck is cumulative inference across a long multi-turn episode.

Evidence link: `docs/log/2026-06-13.md` (TB2-LOCAL-001 section).

## Open questions (not yet tested)

Three levers could change the picture before drawing a broader "local
agents not viable on TB2" conclusion:

1. **Larger per-trial cap.** 1800s is tight for long-context agentic tasks.
   A 4× or 8× cap (120–240 min/trial) would test whether it's a budget
   calibration issue. GPU-days cost is a real concern for a full suite.

2. **Alternate scaffold.** Harbor exposes Claude-Code, OpenHands, and
   Codex CLI adapters. A shorter-context scaffold (e.g., fewer retained
   turns) might reduce per-turn latency and allow completion within the
   current cap.

3. **FT-4B post EXP-013.** qwen3-4b-ft-toolcall generates 2–2.5× faster
   than gemma4-12b on the 3080 Ti. A 4B FT model might complete tasks
   within the 1800s cap even if accuracy is lower. Not yet benchmarked
   on TB2.

## Implication for roadmap

The cloud lane for TB2 anchoring stands (as decided after 2026-06-13).
The local lane for TB2 is blocked by this result **for this specific
configuration** but is not permanently closed. Before labelling local
agents as "not viable on TB2," at least one of the three untested levers
above should be exercised. ADR-015 (SGLang serving) is directly relevant —
higher throughput on small models may unlock lever 3.

## What did not run end-to-end

- Trials 18–89 of the TB2 suite (killed after 17 timeouts to avoid
  GPU-days burn on a known all-zero result).
- Any non-terminus-2 scaffold.
- Any sub-12B model (including FT-4B).
- Any per-trial cap above 1800s.
