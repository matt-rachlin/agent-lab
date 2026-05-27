---
doc_id: model-xlam-2-7b-fc-r
title: salesforce / xlam-2 7b-fc-r
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
# salesforce / xlam-2 7b-fc-r

`litellm_id`: `xlam-2-7b-fc-r` · backend: `llama.cpp` · vram_gb: `5` · context_max: `32768`

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

Source: <https://huggingface.co/Salesforce/xLAM-2-7b-fc-r>
<!-- END AUTOGEN -->

## Description

<!-- BEGIN HAND -->
Salesforce's xLAM-2-7b-fc-r — top-3 on BFCL function-calling benchmark, the lab's
preferred remedy for the R1-distill tool-call structural bugs we saw in F-004 /
F-005. **Status: deferred — no pre-quantized GGUF exists on HF for the 2.x
variant as of 2026-05-27.** A Q2_K v1.x dump exists (`Solshine/xLAM-7b-fc-r-Q2_K-GGUF`)
but that's the older 1.x model at a quant too aggressive for our purposes.

To pull this model we'd need to either:
1. Download the safetensors (`Salesforce/xLAM-2-7b-fc-r`) and quantize locally
   via `llama.cpp/convert_hf_to_gguf.py` + `llama-quantize` — this needs GPU
   compute that Phase 19a is constrained to avoid (EXP-002b is on the GPU).
2. Wait for a community Q4_K_M dump to land (bartowski / unsloth / mradermacher).
3. Run xLAM-2 via vLLM directly from safetensors (slower load, full BF16 in VRAM
   — 14 GB > our 12 GB budget without offload).

Re-evaluate at Phase 19b's llama-swap setup; if still no GGUF by then, drop it
from the registry or schedule a local-quant job after the EXP-002b sweep finishes.

family=mistral-derived fmt=safetensors-needs-conversion source=Salesforce/xLAM-2-7b-fc-r

### VRAM math (Q4_K_M, projected)

- Weights at Q4_K_M (projected): ~4.5 GB
- 7B dense Mistral-derived
- KV cache at 8K ctx (Q8): ~0.4 GB
- Headroom: comfortable on 12 GB when quantized

### Recommended use

- BFCL / function-calling-heavy task suites
- Tool-call structural-fidelity workloads (e.g. as the executor in an
  orchestrator-executor agent split)
- Drop-in replacement when R1-distill produces malformed tool calls

### Throughput expectations

- 50-70 tok/s on RTX 3080 Ti when Q4_K_M GGUF lands (7B dense)
- Cold load: <5 s from NVMe; <1 s from page cache

### License + source

- License: CC-BY-NC-4.0 — research-use only; derived adapters inherit
- Upstream: https://huggingface.co/Salesforce/xLAM-2-7b-fc-r (safetensors)
- GGUF: NONE for 2.x as of 2026-05-27
- Local: not present.

### Verified by lab

Pull failed — no GGUF available, marked deferred. No GPU-using smoke attempted.
Re-check community quant landscape at the start of Phase 19b.
<!-- END HAND -->

## Known issues

<!-- BEGIN HAND -->
- **CC-BY-NC-4.0 license — research use only.** Derived adapters and fine-tunes
  inherit the non-commercial restriction. Flag in Phase 18 (fine-tuning) if
  adapters off xLAM are ever considered.
- **No pre-quantized GGUF for 2.x on HF as of 2026-05-27.** Local quantization
  required, or wait for community dump.
- "fc-r" variant is the regularized function-calling head; the plain `xlam-2-7b`
  is general chat and not the lab's target.
<!-- END HAND -->
