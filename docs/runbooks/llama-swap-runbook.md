---
doc_id: llama-swap-runbook
title: Runbook — llama-swap multi-model orchestrator
zone: lab
kind: runbook
status: active
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
depends_on:
  - doc:lab-phase19-model-orchestration-2026-05-27
tags:
  - lab
  - runbook
  - runbooks
  - orchestration
  - llama-swap
  - phase-19
---

# Runbook — llama-swap multi-model orchestrator

Installed Phase 19b. Sits between LiteLLM (port 4000) and the model
servers; owns VRAM-aware eviction so the big local models don't fight
over the 12 GB on the 3080 Ti.

## Topology

```
agent / sweep cell
       |
       v
   LiteLLM (port 4000)               -- routing, fallbacks, spend tracking
       |
       v
   llama-swap (port 8080)            -- matrix groups, auto load/unload
       |
   +---+----------------------+
   |                          |
   v                          v
 Ollama (port 11434)     llama-server (per-model ports 10001+)
   - small local models    - qwen3-30b-a3b-moe (Q4_K_M MoE, experts on CPU)
   - Ollama Cloud models   - gpt-oss-20b-local (MXFP4, experts on CPU)
                           - phi-4-reasoning-14b (hybrid offload, ngl 25)
                           - hermes-4.3-36b (hybrid offload, ngl 22)
                           - (Phase 19e) llama-3.3-70b-q4
```

## Files

- Binary: `/usr/local/bin/llama-swap` (v217, installed 2026-05-27)
- Config: `/data/lab/code/conf/llama-swap.yaml`
- Service: `/data/lab/services/llama-swap.service`
- LiteLLM bridge: `conf/litellm-config.yaml` — the
  `qwen3-30b-a3b-moe` / `gpt-oss-20b-local` / `phi-4-reasoning-14b` rows
  point at `http://host.containers.internal:8080/v1`.

## Install (one-time)

1. Drop the binary:

   ```bash
   curl -sSL -o /tmp/llama-swap.tgz https://github.com/mostlygeek/llama-swap/releases/download/v217/llama-swap_217_linux_amd64.tar.gz
   tar -C /tmp -xzf /tmp/llama-swap.tgz
   sudo install -m 755 /tmp/llama-swap /usr/local/bin/llama-swap
   ```

2. Install the systemd user unit (the unit file lives in the repo so it
   stays version-controlled, but systemd reads from `~/.config/systemd/user/`):

   ```bash
   ln -sf /data/lab/services/llama-swap.service ~/.config/systemd/user/llama-swap.service
   systemctl --user daemon-reload
   systemctl --user enable --now llama-swap.service
   ```

3. Verify:

   ```bash
   systemctl --user status llama-swap
   curl -s http://localhost:8080/v1/models | jq .
   ```

## Groups (current)

| Group | Members | Behavior |
|---|---|---|
| `small-tools` | `qwen3-reranker-0.6b` | persistent, never evicted by big-model loads |
| `medium-llm` | `qwen3-14b-q4`, `phi-4-reasoning-14b` | exclusive within group, evicts other groups |
| `big-llm` | `qwen3-30b-a3b-moe`, `gpt-oss-20b-local`, `hermes-4.3-36b` | exclusive within group, evicts other groups |
| `ceiling-llm` | `llama-3.3-70b-q4` (llama-swap id; `litellm_id` is `llama-3.3-70b-q4-local`) | exclusive within group; sweep-runner gated via `--allow-slow-models` |

Deferred (model not yet pulled / GGUF not on disk):

- `embedder-big` — `qwen3-embedding-8b-q8`
- (no group) `xlam-2-7b-fc-r` — re-checked 2026-05-27, no upstream GGUF;
  only xLAM-1 (`xLAM-7b-fc-r`) variants exist on the Hub

When any of those GGUFs lands, add an entry under `models:` in
`conf/llama-swap.yaml` and assign it to the appropriate group. The
config has placeholder commentary inline.

## CRITICAL — client-side `keep_alive=0` discipline

The single biggest footgun: Ollama (and llama-server when used through
some clients) honors a per-request `keep_alive` parameter that pins a
model in VRAM for N minutes regardless of llama-swap's group decisions.
If a lab client sends `keep_alive=5m` (the LiteLLM default for our
Ollama entries) to a model behind llama-swap, the group eviction can't
reclaim the slot until that timer expires.

**Rule for all lab code that calls models routed through llama-swap:**

```python
client.chat.completions.create(
    model="qwen3-30b-a3b-moe",
    messages=[...],
    extra_body={"keep_alive": 0},      # <-- always
)
```

Or, when going through LiteLLM:

```python
litellm.completion(
    model="qwen3-30b-a3b-moe",
    messages=[...],
    keep_alive=0,
)
```

Models in `small-tools` (the reranker) are the only exception — they
are persistent by group definition and the keep_alive parameter is
irrelevant for them.

This rule is enforced by review, not by config. If a sweep starts
stalling on cold-loads, grep the agent / sweep code for `keep_alive`
and confirm every call site explicitly sets `0`.

## Operations

```bash
# Status
systemctl --user status llama-swap
journalctl --user -u llama-swap -n 50 --no-pager

# Reload config (the unit runs with -watch-config, so editing the YAML
# is usually enough — but a hard restart is the safe fallback)
systemctl --user restart llama-swap

# Inspect what's loaded right now
curl -s http://localhost:8080/v1/models | jq '.data[].id'

# Force-evict everything (useful before a big sweep)
curl -s -X POST http://localhost:8080/api/unload-all
```

## Adding a new model

1. Pull the GGUF to `/data/models/gguf/<litellm_id>/`.
2. Add a `models:` entry in `conf/llama-swap.yaml` — mirror the existing
   llama-server pattern, set `--model` to the GGUF path, choose
   `-ngl 99` (full GPU) or `-ot 'exps=CPU'` (MoE) or
   `--n-gpu-layers <N>` (hybrid) based on VRAM math.
3. Assign it to a group under `groups:` — never leave a model
   ungrouped if it competes for the same VRAM lane as existing big
   models (an ungrouped model can only run alone, which is correct for
   the `ceiling-llm` 70B case but wrong for everything else).
4. Add the LiteLLM bridge row in `conf/litellm-config.yaml` pointing
   at `http://host.containers.internal:8080/v1`.
5. `systemctl --user restart llama-swap` and verify the new model
   shows up in `/v1/models`.
6. Smoke: send one greedy completion with `keep_alive=0` and confirm
   it loads, responds, and unloads after `ttl`.

## Known issues

- **GPU monitoring**: llama-swap logs
  `[ERROR] failed reading from gpuCh - stopping read goroutine` when the
  NVIDIA driver isn't loaded. Cosmetic — it doesn't affect routing.

## llama.cpp CUDA build (Phase 19d, 2026-05-27)

Phase 19a smokes pushed us off the Vulkan build — on the 3080 Ti, Vulkan
fell over for big-model swaps and we wanted real CUDA throughput. The
current llama.cpp tree at `/data/apps/_vendor/llama.cpp/` is master
commit `66d65ec29` (b8183). We keep two parallel builds:

- `build/`       — original Vulkan-only build (kept as fallback)
- `build-cuda/`  — Phase 19d CUDA build, used by llama-swap

llama-swap macros in `conf/llama-swap.yaml` point at `build-cuda/bin/llama-server`
and set `LD_LIBRARY_PATH=/data/apps/_vendor/llama.cpp/build-cuda/bin` per
model so the bundled `libggml-cuda.so.0` is found at runtime. We did NOT
add `LD_LIBRARY_PATH` to the systemd unit; per-model `env:` blocks in
the YAML are enough because llama-swap propagates them to children.

### CUDA build recipe

```bash
cd /data/apps/_vendor/llama.cpp
# CUDA 12.8 in /usr/local/cuda-12.8 hit the glibc-2.42 noexcept conflict
# (cospi/sinpi/rsqrt redeclared); installed CUDA 12.9:
sudo dnf install -y cuda-nvcc-12-9 cuda-cudart-devel-12-9 cuda-cccl-12-9 \
                    libcublas-devel-12-9 cuda-nvrtc-devel-12-9

# CUDA 12.9 has the same glibc-2.42 conflict — patch crt/math_functions.h
# to add `noexcept (true)` to cospi/sinpi/rsqrt + the float variants. See
# sed in our build log. CUDA 12.9.86 ships with the wrong noexcept on these
# six device-builtin decls on Fedora 43 (glibc 2.42).

# Then configure + build with gcc-14 (gcc-15 unsupported by CUDA):
PATH=/usr/local/cuda-12.9/bin:$PATH cmake -B build-cuda \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-14 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
PATH=/usr/local/cuda-12.9/bin:$PATH cmake --build build-cuda \
  --config Release -j --target llama-server
```

Verify CUDA linkage and device detection:

```bash
ldd /data/apps/_vendor/llama.cpp/build-cuda/bin/llama-server | grep -E 'cudart|cublas'
# Should list libcudart.so.12 and libcublas.so.12 from /usr/local/cuda/...

LD_LIBRARY_PATH=/data/apps/_vendor/llama.cpp/build-cuda/bin \
  /data/apps/_vendor/llama.cpp/build-cuda/bin/llama-server --version
# Header should say:
#   ggml_cuda_init: found 1 CUDA devices:
#     Device 0: NVIDIA GeForce RTX 3080 Ti, compute capability 8.6, VMM: yes
#   version: 8183 (66d65ec29)
```

`~/.local/bin/llama-server` is now a symlink to `build-cuda/bin/llama-server`
(was previously broken; Phase 19c noted this).

### LD_LIBRARY_PATH strategy

llama-swap's systemd unit does NOT export `LD_LIBRARY_PATH` globally. Instead,
each model entry in `conf/llama-swap.yaml` sets it in `env:` (already in
place pre-Phase-19d for the Vulkan build):

```yaml
env:
  - "LD_LIBRARY_PATH=${LD_LIB}"   # where LD_LIB = build-cuda/bin macro
```

Option chosen: Option A's per-process env (via YAML), already in place — we
just changed the macro to point at `build-cuda/bin`. No service file edit needed,
no wrapper script.

### Per-model VRAM tuning (Phase 19d smokes, ctx-size 8192)

The 3080 Ti has 12 GB physical VRAM. The rerank server in `small-tools`
holds ~2.6 GB persistent, leaving ~8.5 GB free for big-model loads. All
n-gpu-layers values below smoked clean with `keep_alive=0` and one
greedy completion through `/v1/chat/completions`:

| Model | Flags | Peak VRAM | First-token latency | Notes |
|---|---|---|---|---|
| `phi-4-reasoning-14b` | `-ngl 25` | ~10.1 GB | ~7 s | Hybrid offload; full -ngl 99 would need 9.1 GB tensors alone and OOMs |
| `qwen3-30b-a3b-moe` | `-ngl 99 -ot 'exps=CPU'` | ~5.6 GB | ~11 s | MoE with experts on CPU — attention layers small, fits fine |
| `gpt-oss-20b-local` (MXFP4) | `-ngl 99 -ot 'exps=CPU'` | ~6.7 GB | ~14 s cold | Also MoE; needs experts-on-CPU pattern. MXFP4 loads natively in b8183 |
| `hermes-4.3-36b` (Q4_K_M) | `-ngl 22 -ctk q8_0 -ctv q8_0` | ~11.4 GB | ~16 s | Tight; only ~600 MB headroom at ctx 8192. Bumping ctx will OOM |

MXFP4 outcome: **loaded natively** — the b8183 CUDA backend supports MXFP4
out of the box, no quant-unknown errors. gpt-oss-20b needs the same MoE
expert-on-CPU pattern as Qwen3-30B-A3B because at full-GPU offload the
MXFP4 tensors alone need ~12 GB (more than free VRAM, not more than
total — the rerank server is the headroom culprit).

### Rebuild trigger

Rebuild when:

- The pinned upstream commit changes (rare — we don't update master often)
- Driver / CUDA toolkit version changes such that linkage breaks
- A new quant format ships (e.g. a Phase 19a-style new model whose GGUF
  uses something newer than b8183)

The Fedora glibc-2.42 patch on `crt/math_functions.h` is sticky to the
CUDA toolkit version. If the dnf transaction reinstalls cuda-cudart-12-9
or cuda-nvcc-12-9, redo the sed. CUDA 13 (when it lands in Fedora) should
ship with `noexcept (true)` on those six decls and drop the patch.

## Phase 19c — `lab.core.model_pool` integration

The lab now ships a thin client (`lab.core.model_pool.ModelPool`) that
sits in front of every sweep cell and agent solve(). It does three
things:

1. **Pre-flight pass** — on `declare(plan)`, fires one `max_tokens=1`
   completion at each model so its GGUF lands in the OS page cache,
   then immediately `POST /api/models/unload/<id>` so the VRAM frees
   for the cell's real work. Cold-NVMe → warm-DDR5 on the second
   load.
2. **Predictive load** — on `step_complete(name)`, fire-and-forget
   warm of the *next* step's first model in a daemon thread.
3. **Explicit eviction** — `teardown()` walks the plan and unloads
   each model. Without this we wait for the per-model `ttl` (default
   600 s) before the slot frees up.

Wired in:

- `lab.sweep.runner.execute_cell(...)` — per-cell, with
  `kb_query` triggering side-models (embedder, reranker).
- `lab.inspect_bridge.solver.model_with_tools(...)` — per-turn
  inside the multi-turn agent loop.
- CLI: `lab agent run --pipeline-plan-only` prints the JSON plan that
  would be declared, without loading anything.

Network failures inside the pool are logged and swallowed. The
inference call later in the pipeline will trigger its own load if
llama-swap is down, just paying the cold-cache penalty.

### CUDA fragmentation discipline

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` everywhere
PyTorch is involved. This is in `.env.example` and the lab's `.env`
(loaded by `pydantic-settings` in `lab.core.settings` via SettingsConfigDict).
Ollama and llama.cpp don't use PyTorch — they're pure C++ + GGUF mmap —
so the variable is a no-op for those processes. It matters for:

- the Phase 7 reranker host service (sentence-transformers / torch)
- any future fine-tuning code path (Phase 18)
- ad-hoc `lab/scripts/*.py` that does `from_pretrained(...)`

Without it, hot-swap between big PyTorch models eventually fragments
the allocator and triggers an OOM on the next load.

If you ever do a raw `torch.load(...)` or `from_pretrained(...)` in a
sub-agent (i.e. NOT through llama-swap/Ollama), apply the eviction
discipline on the way out:

```python
del model              # drop the Python ref so the C++ destructor fires
torch.cuda.empty_cache()
gc.collect()
```

Skipping this on a 30B-class model leaves multiple GB pinned in the
PyTorch allocator's free list, invisible to `nvidia-smi`'s "used" but
unavailable to the next model. ModelPool already handles llama-swap-
managed models; this discipline is only relevant for raw-torch sites.

## Phase 19e — 70B quality-ceiling lane

Added 2026-05-27. Llama-3.3-70B-Instruct Q4_K_M served via llama.cpp
hybrid GPU+CPU offload as the lab's **offline-only** quality ceiling.
Not a swap-in for any interactive workload — runs ~1.8 tok/s on this
hardware (single-digit, where the medium models do 30-60 tok/s).

### Working config

```yaml
"llama-3.3-70b-q4":
  cmd: |
    ${LLAMA_SERVER}
    --port ${PORT} --host 127.0.0.1 --ctx-size 8192
    --n-gpu-layers 14 -ctk q8_0 -ctv q8_0
    --model ${MODELS_DIR}/llama-3.3-70b-q4/Llama-3.3-70B-Instruct-Q4_K_M.gguf
    --jinja
  ttl: 1800
  groups: [ceiling-llm]
```

### Working `--n-gpu-layers` and why it differs from the plan

Plan §19e called for ngl=21 (per research-best on a clean 12 GB card).
The Phase 19e tuning smoke had:

| ngl | Outcome | Detail |
|-----|---------|--------|
| 21 | OOM | `cudaMalloc 10846 MiB > free 8.5 GiB` — research assumed 12 GB total free; rerank-server-resident (~2.6 GB) cuts it to 8.5 GB |
| 15 | OOM | weights fit (7968 MiB) but KV-cache q8_0 alloc (238 MiB) tipped it over |
| **14** | **works** | weights 7445 MiB on GPU, KV+scratch take total to 11.7 GB / 12 GB; 219 MB headroom |

The 12 GB physical VRAM minus persistent rerank server (~2.6 GB) is the
hard constraint here. We could squeeze ngl up by evicting the rerank
server, but the cost is enormous re-load for every kb_query downstream;
ngl=14 with the rerank server resident is the better lab-wide trade.

### Measured smoke (2026-05-27)

PBS-Agent task `fs-read-and-copy` end-to-end via
`lab agent run --model llama-3.3-70b-q4-local --allow-slow-models`:

| Metric | Value |
|---|---|
| Total wall (cold load + 3 turns) | **103 s** |
| In-inference latency (3 turns) | 101 s |
| Peak VRAM during run | **11.8 GB** (~470 MB headroom on the 12 GB card) |
| Turns used | 3 |
| Tool calls | 2 |
| Generation throughput (single-turn measured) | **~1.83 tok/s** (147 tokens in 80 s, ctx 56) |
| Prompt throughput | ~5.5 tok/s (prompt processing) |
| `end_state` scorer | **1.0** (PASS) |
| `tool_correctness` scorer | 1.0 |
| `budget_respected` scorer | 1.0 |

The 1.8 tok/s is slower than the plan's 6-10 tok/s aspiration. Root
cause: only 14/81 layers run on GPU (vs the 21/81 the research assumed
was possible). The remaining 67 layers are CPU-bound; aggregate
throughput is dominated by the CPU pipeline. A future fix is to free
the rerank server's VRAM during ceiling-llm sweeps (would need cross-
group eviction in llama-swap config OR a wrapper that pauses the
persistent reranker) — left for future work; today's plain ceiling-llm
group doesn't do that.

### Sweep-runner `--allow-slow-models` gate

`llama-3.3-70b-q4-local` is registered in `lab.models` with `capabilities`
containing `slow_mode`. `lab sweep run` and `lab agent run` refuse to
dispatch with this model unless the operator passes
`--allow-slow-models`. The DB lookup is in
`lab.sweep.runner._slow_models_in`; the gate raises `SlowModelGateError`
in the sweep runner and `typer.Exit(2)` in the agent CLI. Adding future
ceiling-mode models is a registration concern only — no code change.

```bash
# This refuses (default-safe):
lab sweep run conf/sweep/some-sweep-with-70b.yaml --dry-run
# ERROR: sweep references slow_mode (ceiling-class) models without
# --allow-slow-models: ['llama-3.3-70b-q4-local']. These run 6-10 tok/s; pass
# the flag explicitly to opt in, or drop them from spec.models.

# Same sweep with explicit opt-in runs:
lab sweep run conf/sweep/some-sweep-with-70b.yaml --allow-slow-models
```

The agent CLI surface:

```bash
lab agent run --suite pbs-agent-v0.1 --task fs-read-and-copy \
  --model llama-3.3-70b-q4-local --allow-slow-models
```

### LiteLLM route

`llama-3.3-70b-q4-local` (note: the `-local` suffix distinguishes the
LiteLLM model_name from the llama-swap internal ID). Timeout 1800 s
because a multi-turn agent task at ~1.8 tok/s can stretch to multiple
minutes per turn; 1800 s covers the worst-case full-budget task.

### What to do when it OOMs

If a different lab process has parked memory on the GPU (Chrome,
Loupe, an old PyTorch session), `ngl 14` may still fail. Steps:

```bash
# Check who has VRAM
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Evict whatever you can; the 70B needs ~8 GB free over the rerank server's footprint
# If you can't free enough, drop --n-gpu-layers further (each layer is ~530 MiB)
```

The cmd is tuned for the steady-state lab box. Drift here means a
PyTorch session leaked or Chrome went heavy; fix that, don't re-tune
the model.

## Rollback

If llama-swap misbehaves, revert LiteLLM to direct Ollama for the
Phase 19a models (they have Ollama tags too, just not currently exposed):
edit `conf/litellm-config.yaml`, change the three new entries' `api_base`
back to `http://host.containers.internal:11434` and `model:` back to the
`ollama_chat/<tag>` form, then `systemctl --user stop llama-swap`.
