---
doc_id: sglang-phase1-integration
title: 'SGLang Phase 1 integration spec (Stage A + B APPLIED + VALIDATED)'
zone: lab
kind: spec
status: active
owner: m
created: '2026-06-14'
last_updated: '2026-06-14'
last_verified: '2026-06-14'
tags: [lab, spec, sglang, serving, integration, applied]
---
# SGLang Phase 1 integration (DRAFT v2)

## STAGE B — APPLIED + VALIDATED (2026-06-14, commits 77270f6 + c107b49)

The coordinated runner change shipped and was GPU-validated end to end.

- **G1/M1/M2 (runner, 77270f6):** `_resident_batch_model` gates an opt-in
  throughput path (sglang-local + single model + `max_concurrency>1`); cells
  dispatch through a `ThreadPoolExecutor` (G1) under ONE batch `gpu_lease` with
  `skip_cell_lease=True` per cell (M1); `ModelPool.declare(preflight=False)` once
  + `teardown()` once, each cell `model_pool=None` (M2). 14 new unit tests
  (eligibility, lease-plumbing, one-lease-per-batch, lifecycle, serial/concurrent
  parity); mypy --strict + pyright green. Serial/ollama/cloud/c1 paths unchanged.
- **Serving tuning (c107b49):** GPU validation exposed two serving-layer issues
  the runner change surfaced (not caused). (1) `mem-fraction-static 0.55` (the G3
  measure-first value, chosen for a rerank the spec believed dead — it is ACTIVE)
  starved SGLang's KV under c16 -> a 429. (2) LiteLLM's router amplified that ONE
  429 into a 60s single-deployment cooldown -> 501/512 cells failed. Fixes: bump
  to `0.70` (KV for ~32 concurrent BFCL reqs; coexists with the 2.6 GB rerank,
  8.4+2.6<12) + `--max-running-requests 32`; `RateLimitErrorAllowedFails: 1000`
  on the router so a transient 429 can't blackout the lone awq deployment. Also
  resolved the qwen3 thinking knob for the SGLang arm: `chat_template_kwargs.
  enable_thinking=false` in the cell `extra` (BFCL tasks own their system prompt,
  so `/no_think` isn't injectable via a system prompt; this is the only config-
  expressible knob, forwarded verbatim by `call_litellm_chat`).
- **Results (512-cell BFCL v3, N=16, warm):** c16 = **271s** wall-clock, pass
  **0.838** (cohort-comparable: gemma4 0.85, qwen3-14b 0.906 -> AWQ-W4A16 NOT
  degraded), 1 transient 429 (->0 on resume). c32 = 272s, 22 transient 429s — NO
  throughput gain (path saturates at ~c16; c16 is the production setting). vs the
  serial Ollama baseline (D5-BASELINE-BFCL-001, 512 cells = 2419s): **~8.9x**.
  Zero cooldown cascades after the litellm fix; no OOM at 0.70.
- **Acceptance:** A1 ok (concurrent, one lease, resident) · A2 ok (pass not
  degraded — the spec's "0.41" figure predated the thinking fix) · A3 ok (no OOM
  measured at 0.70 alongside the active rerank) · A4 ok (271s << 2419s).
- **Follow-ups (not blocking):** transient-429 ramp errors (~1 at c16) are mopped
  by `--resume`; a runner-side 429 backoff/retry on the resident-batch path would
  make single-pass runs error-free (deferred). c32 offers no gain -> keep c16.

Original DRAFT v2 spec text (pre-application) preserved below for provenance.

---
# SGLang Phase 1 integration (DRAFT v2)

Status: DRAFT, reviewed once (adversarial). Phase 0 = **GO** (7.1x tok/s @ c32 vs
Ollama-c1, fits 12 GB standalone). **Not applied** — live changes to
`conf/llama-swap.yaml`, `conf/litellm-config.yaml`, `lab.models`, and the sweep
runner are gated on Matt's go.

## REVISED SCOPE (the headline correction)
ADR-015 §4 ("Sweep runner untouched; `-awq` sweeps raise `max_concurrency`") is
**wrong**. Verified against the code, Phase 1 is a **coordinated runner change**,
not config-only. The four code changes below are interdependent and must land
together for any throughput win:
- **G1** add concurrent cell dispatch (today the run loop `runner.py:1649` is
  strictly serial; `max_concurrency` is read NOWHERE — dead config).
- **M1** redesign the GPU lease for SGLang. The lease (`lab:gpu:lease`, single
  global `SET nx`) is acquired/released **per cell** (`runner.py` 821/1027). With a
  thread pool, the first cell holds it and the rest block — concurrency is defeated
  unless the SGLang batch holds **one lease for the whole resident run**.
- **M2** move model lifecycle out of the cell loop. `model_pool.declare()` runs at
  the start of every cell (`runner.py:662`) and `teardown()` in every cell's
  `finally` (759). For SGLang's ~5-min CUDA-graph startup this is fatal (reload per
  cell). Declare once before the loop, teardown once after (the ModelPool is already
  sweep-scoped at 1612); plumb a "skip per-cell teardown" flag through `execute_cell`.
- **B1** make backend predicates SGLang-aware (below).

## BLOCKERS / GAPS (consolidated, post-review)
- **B1 (blocker) — backend-string predicates are hardcoded to `ollama-local`.**
  `_is_local_backend()` (`runner.py:391`) returns `backend == "ollama-local"`. An
  `sglang-local` arm therefore (a) runs **without `gpu_lease`** (no VRAM mutual
  exclusion — breaks G3), and (b) at `runner.py:1017` gets `tool_choice="required"`
  (since `"ollama"` not in `"sglang-local"`). FIX: teach `_is_local_backend` (and
  the agent solver branch) that `sglang-local` is a llama-swap-routed local backend
  so it acquires the lease. For tool_choice: `required` is the F-017-correct choice
  **iff SGLang honors it** — VERIFY SGLang serves `tool_choice="required"` with
  `--tool-call-parser qwen`; if not, route SGLang to `auto`.
- **B2 (major) — agent path also branches on `== "ollama-local"`** (`solver.py:678`).
  v1 scope is single-turn BFCL ONLY; explicitly keep the `-awq` arm off the agent
  path until validated. Nothing enforces this today — state it in the sweep scope.
- **G3 (VRAM) — coexistence, revised.** The rerank server (`qwen3-reranker-0.6b`,
  `persistent`, ~2.6 GB) is **currently inactive** (`rerank.service` dead), so its
  residency is conditional on the systemd unit. When up: SGLang mem-fraction 0.80
  (~11.6 GB) + 2.6 GB = OOM. The repo's llama.cpp notes repeatedly warn of ~2.6 GB
  hidden CUDA-context that `nvidia-smi` free doesn't show — so **mem-fraction 0.62
  is optimistic**. RESOLUTION: measure at **0.55** first; OR adopt option (b) — evict
  the rerank on `-awq` load (mirror the llama-3.3-70b ceiling-wrapper). Decide by
  measurement, not estimate.
- **G4 (correctness gate) — RESOLVED + extended.** `--tool-call-parser qwen` is a
  valid choice in the pinned image (verified via `--help`; no `qwen3`; `qwen25` is
  2.5-specific, `qwen3_coder` is coder-specific). Phase-0 `pass_rate=0` was the
  MISSING parser. Acceptance: serve with `--tool-call-parser qwen`, re-run the c1
  BFCL pass check, require pass ≈ the Ollama-Q4 baseline (**0.41**). ALSO validate
  the tool_choice path from B1 (don't assume the parser flag alone fixes it).

## The five seams (corrected)

### Seam 1 — `conf/llama-swap.yaml`
New foreground-launched container model (custom podman root so the image resolves;
`--network=host` for the port-publish gotcha; pre-start `rm -f` guard against an
orphaned host-net container holding the port):
```yaml
  "qwen3-4b-awq":
    cmd: |
      sh -c 'podman --root /data/lab/containers/storage --runroot /data/lab/containers/run rm -f sglang-qwen3-4b-awq 2>/dev/null;
      exec podman --root /data/lab/containers/storage --runroot /data/lab/containers/run
      run --rm --name sglang-qwen3-4b-awq --device nvidia.com/gpu=all --ipc=host --network=host
      -v /data/lab/models/awq/qwen3-4b-awq:/model:ro,Z docker.io/lmsysorg/sglang:latest
      python3 -m sglang.launch_server --model-path /model --host 127.0.0.1 --port ${PORT}
      --served-model-name qwen3-4b-awq --mem-fraction-static 0.55
      --attention-backend flashinfer --tool-call-parser qwen'
    cmdStop: |
      podman --root /data/lab/containers/storage --runroot /data/lab/containers/run stop -t 10 sglang-qwen3-4b-awq
    name: "Qwen3-4B AWQ (SGLang, throughput tier)"
    description: "In-house AWQ-W4A16; continuous batching for small-model sweeps."
    ttl: 600
```
New exclusive group `sglang-awq` (`swap: true, exclusive: true`). VERIFY: the pinned
llama-swap version actually invokes `cmdStop` on swap/TTL (not just SIGKILL of the
foreground PID); confirm `healthCheckTimeout: 500` tolerates a cold ~5-min capture.

### Seam 2 — `conf/litellm-config.yaml`
Mirror `qwen3-30b-a3b-moe` EXACTLY (verified): `openai/` provider + llama-swap proxy
+ `timeout: 600` (the existing routes set it; v1 draft had omitted it):
```yaml
  - model_name: qwen3-4b-awq
    litellm_params:
      model: openai/qwen3-4b-awq
      api_base: http://host.containers.internal:8080/v1
      api_key: "dummy"
      timeout: 600
```
Not subject to the ollama keep_alive preflight (that fires only on `ollama_chat/`).

### Seam 3 — `lab.models`
`register.py` has CLOUD_MODELS + an Ollama-derived path; **no curated non-ollama-local
list** (confirmed) — add one (e.g. `SGLANG_MODELS`), idempotent via
`ON CONFLICT (litellm_id)`. Row: `backend=sglang-local litellm_id=qwen3-4b-awq
quant=awq-w4a16 source_sha256=630180951e... vram_gb~=7 context_max=40960
output_max=4096 capabilities=[tool_call] ollama_tag=None`. NOTE: `sync_models()` only
counts ollama-local/cloud (line 286) — extend the summary or accept under-count.
CORRECTION: the comparison is **AWQ-W4A16 (SGLang) vs Q4_K_M (Ollama qwen3:4b)** — a
4-bit quant-method + engine comparison; there is NO llama-swap GGUF `qwen3-4b` arm.
(`qwen3-4b-ft-toolcall-q4` is the other Ollama-Q4 point.)

### Seam 4 — manifest provenance
`register.py` reads `/data/lab/models/awq/qwen3-4b-awq/MANIFEST.json`
(`output_sha256 630180951e...`) → `source_sha256` + notes (recipe/calib). Verified.

### Seam 5 — sweep config (additive new file)
`conf/sweep/cand-cap-qwen3-4b-awq.yaml`: same 32-task BFCL / N=16, `models:
[qwen3-4b-awq]`, `max_concurrency: 16`, single-turn ONLY (B2). Additive; no existing
config touched.

## Acceptance criteria
- A1: `-awq` sweep at `max_concurrency>=16` dispatches concurrently (G1) under ONE
  lease for the resident batch (M1) with SGLang resident across the batch (M2).
- A2: pass ≈ 0.41 Ollama-Q4 baseline on the c1 reference (G4 + B1 tool_choice).
- A3: no OOM alongside the rerank (G3) — measured, not estimated.
- A4: 512-cell `-awq` sweep wall-clock materially below the Ollama equivalent
  (Phase-2 re-baseline; report util + wall delta).

## Rollback
Additive arm/route/group + one sweep file; runner changes gated behind
`max_concurrency>1` + `backend==sglang-local` (default 1 / ollama = unchanged for
every existing sweep). No existing arm modified (ADR-015 §3). Revert = remove the
additions; the new code paths go dormant.

## Review log
- Round 1 (adversarial, 2026-06-14): found B1 (lease + tool_choice predicate), B2
  (agent-path predicate), M1 (lease serializes concurrency), M2 (lifecycle is
  cell-scoped, not hook-expressible), corrected the GGUF-baseline premise, lowered
  mem-fraction to 0.55 (hidden CUDA context), resolved parser to `qwen`, added the
  litellm `timeout` and the port-orphan guard. Scope revised from "config-only" to
  "coordinated runner change."

## Open questions
- Does SGLang honor `tool_choice="required"` with the qwen parser? (decides B1).
- G1 scope: minimal single-turn concurrent dispatcher vs general runner concurrency
  (v1 = single-turn `-awq` only).
- M1 lease: per-batch lease token vs a backend-scoped lease bypass — design before
  coding.
- llama-swap `cmdStop` invocation semantics on the pinned version.
