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
   - Ollama Cloud models   - gpt-oss-20b-local (MXFP4)
                           - phi-4-reasoning-14b
                           - (future) hermes-4.3-36b, llama-3.3-70b-q4
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
| `big-llm` | `qwen3-30b-a3b-moe`, `gpt-oss-20b-local` | exclusive within group, evicts other groups |

Deferred (model not yet pulled / GGUF not on disk):

- `ceiling-llm` — `llama-3.3-70b-q4` (Phase 19e)
- `embedder-big` — `qwen3-embedding-8b-q8`
- `big-llm` member: `hermes-4.3-36b` (download was interrupted)
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
- **llama.cpp build is Vulkan-only**: the binary at
  `/data/apps/_vendor/llama.cpp/build/bin/llama-server` was compiled
  without CUDA. With the NVIDIA driver loaded it falls back through
  Vulkan, which on a 3080 Ti is ~10-20% slower than CUDA but works.
  If we hit throughput problems, rebuild llama.cpp with `-DGGML_CUDA=ON`.
- **`~/.local/bin/llama-server` symlink is broken** — points at
  `~/applications/llama.cpp/build/bin/llama-server` which doesn't exist.
  llama-swap.yaml uses the real path under `/data/apps/_vendor/`
  directly. Don't rely on the symlink.

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

## Rollback

If llama-swap misbehaves, revert LiteLLM to direct Ollama for the
Phase 19a models (they have Ollama tags too, just not currently exposed):
edit `conf/litellm-config.yaml`, change the three new entries' `api_base`
back to `http://host.containers.internal:11434` and `model:` back to the
`ollama_chat/<tag>` form, then `systemctl --user stop llama-swap`.
