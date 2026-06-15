---
doc_id: f-012-agentic-tool-calling-failure-modes
title: 'F-012: EXP-007/008 — agentic tool-calling fidelity, not size or coding
  skill, gates local coding-agent performance. Three failure modes: text-emitted
  tool calls, premature chain termination, plan narration. A 12B generalist beats
  models 2–6× its size; a format-fallback shim recovers mechanics but not scores.'
zone: lab
kind: finding
status: active
owner: m
created: '2026-06-11'
last_updated: '2026-06-11'
last_verified: '2026-06-11'
depends_on:
- kind: doc
  target: exp-007
- kind: doc
  target: exp-008
- kind: doc
  target: f-005
- kind: code
  target: lab:packages/lab-inspect/src/lab/inspect_bridge/solver.py
- kind: artifact
  target: 's3://lab trajectory JSONL: CODER-BENCH-001, LLAMA-70B-AGENT-BENCH-001, FALLBACK-TEST, HARD-BENCH-001'
tags:
- lab
- finding
- findings
- agentic
- tool-use
- failure-modes
- confidence-high
- importance-8
---

# F-012: Agentic tool-calling fidelity gates local coding-agent performance

## TL;DR

**On the lab's agentic suites, model size and coding-leaderboard strength
do not predict agent performance; protocol fidelity does.** gemma4-12b
(generalist) scores 1.000 on pbs-agent-v0.1 and 0.938 on the hard suite,
beating Llama-3.3-70B (0.583), Qwen2.5-Coder-32B (0.000), and
Devstral-24B (0.375 v1 / 0.531 v2). Trajectory-level inspection
(MLflow Tracing + JSONL in MinIO) attributes every losing model's gap to
one of three failure modes, none of which is "wrote bad code":

1. **Text-emitted tool calls** — the model writes
   `{"name": "http_fetch", "arguments": {...}}` as message *content*
   with no structured `tool_calls` field; the harness correctly reads it
   as a final answer. Qwen2.5-Coder-32B does this on every task (hence
   0.000); Llama-3.3-70B intermittently. A dedicated chat template with
   tool handlers did not change the rate.
2. **Premature chain termination** — Llama-3.3-70B passes short chains
   (including one 4-tool sequence) but goes 0/3 seeds on every task that
   requires continuing *after* a tool result: it fetches or executes
   once, then writes a summary paragraph instead of the next call.
3. **Plan narration** — Devstral-24B replies with a friendly plan and
   markdown *pseudo-code* of the calls it would make, executing nothing;
   the episode ends on the spot. Prompt-sensitive: see
   [F-013](F-013-prompt-robustness-model-property.md).

## The shim test: format is not the disease

A harness fallback (`_extract_text_tool_calls`, solver.py, 600b7a8)
parses content-embedded JSON and executes it anyway. Mechanics
recovered — Qwen2.5-Coder went from 1 to 6 executed calls on the probe
task — **scores did not move.** The recovered calls contain placeholder
pseudo-code (`content = $response['content']`) presuming a
variable-binding REPL that a stateless tool loop doesn't provide. The
model isn't formatting its actions wrong; it's reasoning in a different
execution model. Patching the envelope exposes the next layer of the
same training gap. The fallback stays in the harness as a diagnostic,
not a rescue.

## Why the winners win

gemma4-12b and Qwen3-Coder-30B emit clean structured `tool_calls` every
turn and consume tool results correctly. Qwen3-Coder is explicitly
agentic-trained; gemma4's release likely included similar data.
Qwen2.5-Coder-32B is a strong *completion* model — its HumanEval-class
scores are real — but operating a tool loop is a different trained
behavior, and that distinction is invisible on static coding
leaderboards.

## Caveats

- Hard-suite numbers are single-seed (EXP-008); the N=8 confirmation is
  EXP-009/HARD-BENCH-003. The CODER-BENCH-001 extremes (1.000/0.000 × 3
  seeds × 12 tasks) are not in doubt.
- All models quantized via Ollama defaults (Q4-class); quantization as a
  contributor was not isolated, though Qwen2.5's failure being *total
  and format-shaped* makes it an unlikely primary cause.

## Consequences

- gemma4-12b is the lab's local coding-agent default.
- Model selection for agent work must screen on tool-loop fidelity
  first; coding-leaderboard rank is not a proxy.
- Public writeup: `docs/writeups/local-coding-agent-benchmark.md`.
trust_level: unverified
