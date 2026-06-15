---
doc_id: model-pull
title: 'SOP: Registering a model (local or cloud)'
zone: lab
kind: guide
status: active
owner: m
created: '2026-05-25'
last_updated: '2026-05-25'
last_verified: '2026-05-25'
tags:
- lab
- guide
- sops
---
# SOP: Registering a model (local or cloud)

**Purpose:** add a new model to the lab's catalog and route it through LiteLLM.

---

## Local model (via Ollama)

1. **Pull the GGUF** with the canonical Ollama tag:

   ```bash
   ollama pull qwen3:14b-q4_K_M
   ```

   Verify pull-time digest matches Ollama Hub. Capture the tag exactly (`name:variant[-quant]`) — this is the source of truth.

2. **Add to `conf/serving/litellm-config.yaml`** with a stable `model_name`:

   ```yaml
   - model_name: <canonical-id>          # e.g. qwen3-14b-q4
     litellm_params:
       model: ollama_chat/<ollama-tag>
       api_base: http://host.containers.internal:11434
   ```

   Conventions for `model_name`:
   - lowercase, hyphenated
   - drop quant when it's the default Q4_K_M; include Q5/Q8/fp16 explicitly
   - drop `instruct` (assumed unless noted)
   - matches the friendly map in `lab.models.register._FRIENDLY`

3. **Restart LiteLLM** so the proxy picks up the new model:

   ```bash
   cd /data/lab/services && podman-compose restart litellm
   ```

   Wait ~10s for healthcheck.

4. **Register in the lab DB**:

   ```bash
   uv run lab models register      # walks `ollama list` + appends cloud catalog
   # (alias for: uv run python -m lab.models.register)
   ```

5. **Smoke-test the route**:

   ```bash
   curl -s -H "Authorization: Bearer $(cat /data/lab/services/litellm-master-key)" \
     -H "Content-Type: application/json" \
     -X POST http://localhost:4000/v1/chat/completions \
     -d '{"model":"<model_name>","messages":[{"role":"user","content":"ready?"}],"max_tokens":50}'
   ```

   Expect a non-empty response. If empty, common causes:
   - Reasoning-mode model (qwen3) with `max_tokens` too small — increase.
   - VRAM contention — Ollama can't load the new model; check `ollama ps`.

## Cloud model (Ollama Cloud Pro)

Cloud models are proxied transparently by the local Ollama daemon via the user's ed25519 signin. **No `OLLAMA_API_KEY` is required in the container** — see [ADR-002](../adr/ADR-002-inference-routing.md).

1. **Verify the tag exists** in the cloud catalog:

   ```bash
   ollama run <model>:cloud "hello"
   ```

2. **Add to `conf/serving/litellm-config.yaml`** with a `-cloud` suffix on the model_name:

   ```yaml
   - model_name: <canonical-id>-cloud      # e.g. gpt-oss-120b-cloud
     litellm_params:
       model: ollama_chat/<model>:cloud
       api_base: http://host.containers.internal:11434
       max_tokens: 16384                    # Ollama Cloud output cap
       timeout: 600
   ```

3. **Append to `CLOUD_MODELS` list** in `src/lab/models/register.py` with publisher/variant/license/capabilities/notes fields.

4. Restart LiteLLM, register, smoke-test as above.

## Quality checklist for any new model row

Before using a model in a sweep claim, verify:

- [ ] `vram_gb` is filled (local only)
- [ ] `context_max` and `output_max` reflect the actual tag
- [ ] `capabilities` array includes truthful tags (`tool_call`, `vision`, `reasoning`, `json`)
- [ ] `license` is recorded
- [ ] `notes` mentions any quirks (reasoning-by-default, output cap, etc.)
- [ ] A 1-token round-trip succeeds via LiteLLM
- [ ] A 1024-token round-trip succeeds (catches reasoning-mode budget issues)

## Retiring a model

```sql
UPDATE models SET retired_at = NOW() WHERE litellm_id = '<id>';
```

Retired models stay in the DB so historical `experiment_runs` rows still join cleanly.

## Storage management

Per `~/.claude/CLAUDE.md`: `/data` is RAID0 (no redundancy). Ollama stores GGUFs under `~/.ollama/models/` by default (which lives on `/home`, redundant). Don't move models to `/data/lab/models/` unless they're cheaply re-downloadable.

When the home disk fills:

```bash
ollama rm <tag>          # remove a model not currently used
```

Then update the `models` row: `UPDATE models SET retired_at = NOW() WHERE ollama_tag = '<tag>'`.
