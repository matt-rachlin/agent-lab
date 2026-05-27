---
doc_id: model-llama-3-3-70b-q4
title: meta / llama-3.3 70b
zone: lab
kind: card
status: active
owner: m
created: 2026-05-27
last_updated: 2026-05-27
last_verified: 2026-05-27
tags:
- lab
- card
- model-cards
- phase-19a
---

<!-- BEGIN AUTOGEN -->
# meta / llama-3.3 70b

`litellm_id`: `llama-3.3-70b-q4` · backend: `llama.cpp` · vram_gb: `40` · context_max: `131072`

## Usage

No `experiment_runs` rows in the last 30 days.

## Performance (lifetime aggregate)

- runs: 0 (done=0, error=0)
- mean latency: n/a
- mean tokens_in: n/a
- mean tokens_out: n/a
- mean cost: n/a

## References

No findings cite this model yet.

Source: <https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF>
<!-- END AUTOGEN -->

## Description

<!-- BEGIN HAND -->
Meta's Llama-3.3-70B-Instruct quantized to Q4_K_M — the lab's offline-only
**quality ceiling** model. Mode: `offline-only`. Per Phase 19 strategic
decisions: 6-10 tok/s with hybrid GPU+CPU offload, useful only for
non-interactive batch eval. Tagged `slow_mode` in `lab.models.capabilities`
so sweep runners and `lab agent run` refuse to use it unless the
`--allow-slow-models` flag is set explicitly. Do not use in interactive
cells; sweep-runner gates via `--allow-slow-models`.

family=llama3 fmt=gguf source=bartowski/Llama-3.3-70B-Instruct-GGUF

### VRAM math (Q4_K_M hybrid offload, Phase 19e working config)

- Weights on disk: ~39.6 GB Q4_K_M GGUF
- 70B dense — does NOT fit in 12 GB even quantized
- Working `--n-gpu-layers`: **14** (validated Phase 19e smoke 2026-05-27).
  Plan called for 21, but the rerank server (~2.6 GB persistent in
  `small-tools`) cut effective free headroom to 8.5 GB; ngl=21 OOM'd
  with `cudaMalloc failed: 10846 MiB > free`, ngl=15 OOM'd on the q8 KV
  alloc, ngl=14 fits with ~219 MB headroom on the steady-state lab box.
- ctx-size 8192, Q8 KV cache (`-ctk q8_0 -ctv q8_0`) — halves KV footprint
  at ~3 % perplexity cost.
- Peak VRAM during smoke: **11.8 GB** (~470 MB headroom)
- DDR5 resident: ~30 GB (remaining 67/81 layers as CPU offload via mmap)

### llama-swap config

Group: `ceiling-llm` (exclusive — when this loads, all `big-llm` members
are evicted). TTL 1800s (longer than other big models because the cold
load from NVMe is ~60-90 s; we want to keep it resident across an offline
batch). See `conf/llama-swap.yaml` for the canonical command.

### LiteLLM route

`llama-3.3-70b-q4-local` → llama-swap on port 8080 → llama-server. Timeout
1800 s (single turn can take 100-170 s; a multi-turn agent task can stretch
to multiple minutes per turn). Defined in `conf/litellm-config.yaml`.

### Recommended use

- Offline batch reference cells (the "what if we had unlimited budget" baseline)
- Pre-registered comparison runs where ceiling is the question (EXP-006b)
- NOT for: interactive workflows, latency-sensitive cells, sweeps without
  `--allow-slow-models`

### Throughput expectations

- **Measured Phase 19e smoke: ~1.83 tok/s** (slower than the plan's
  research-cited 6-10 tok/s, because only 14/81 layers run on GPU here
  vs the 21/81 the research assumed). Aggregate is CPU-bound.
- Prompt processing: ~5.5 tok/s
- Cold load: 60-90 s from NVMe; ~10 s from page cache
- A typical PBS-Agent task (~1024 output tokens) = ~9-10 minutes at
  steady state, plus cold-load on first cell of a batch

### License + source

- License: Llama 3.3 community — commercial OK with restrictions (no training
  competing models on outputs, attribution required)
- Upstream: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct (gated upstream;
  bartowski's GGUF dump is unauth-accessible)
- Local: `/data/models/gguf/llama-3.3-70b-q4/Llama-3.3-70B-Instruct-Q4_K_M.gguf`

### Verified by lab

Phase 19e formal smoke 2026-05-27: `fs-read-and-copy` PBS-Agent task
end-to-end via `lab agent run --model llama-3.3-70b-q4-local
--allow-slow-models`. Result: `end_state` = 1.0, `tool_correctness` =
1.0, `budget_respected` = 1.0 (full pass). 3 turns, 2 tool calls,
total wall 103 s, peak VRAM 11.8 GB. See the Phase 19e section of
`docs/runbooks/llama-swap-runbook.md` for the recorded latency / VRAM
table.
<!-- END HAND -->

## Known issues

<!-- BEGIN HAND -->
- **Hybrid offload only.** ~1.8 tok/s measured wall throughput (slower
  than the research-cited 6-10 tok/s on a clean 12 GB card — see runbook
  for why); a PBS-Agent-style multi-turn task can run minutes per turn.
- **--n-gpu-layers: 14 steady-state, 16 with `ceiling-sweep-wrapper.sh`.**
  The repo's checked-in value in `conf/llama-swap.yaml` is **16** (the
  ceiling-mode value); steady-state runs without the wrapper must lower
  it back to 14 or the load OOMs because the rerank server (~2.6 GB
  persistent) eats the headroom. ngl=21 (the Phase 19e plan target)
  doesn't fit even with the wrapper because llama.cpp reserves ~2.6 GB
  of CUDA context that `nvidia-smi --memory.free` doesn't show. Measured
  throughput at ngl=16 is **1.92 tok/s** (vs 1.83 at ngl=14, +5 %); the
  wrapper's main value is operational predictability rather than raw
  speed. Auto-split heuristics can mis-allocate, so always pass the
  explicit flag.
- Llama 3.3 community license — commercial OK with restrictions (no
  training-on-outputs to make competing LLMs).
- Cold-load from NVMe is ~60-90 s (40 GB read); page-cache hit drops to ~10 s
  but it'll evict everything else.
- **Tight VRAM headroom (~470 MB at smoke).** If a parked PyTorch
  session or Chrome bloat steals additional VRAM, the load OOMs. See
  runbook "what to do when it OOMs".
<!-- END HAND -->
