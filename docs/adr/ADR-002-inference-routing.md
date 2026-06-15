---
doc_id: adr-002-inference-routing
title: 'ADR-002: Inference routing — LiteLLM proxy over local Ollama daemon'
zone: lab
kind: adr
status: superseded
owner: m
created: '2026-05-25'
last_updated: '2026-06-14'
last_verified: '2026-05-25'
tags:
- lab
- adr
---
# ADR-002: Inference routing — LiteLLM proxy over local Ollama daemon

Status: superseded by ADR-018 (2026-06-14)
Date: 2026-05-25
Deciders: Matt Rachlin

## Context

The lab compares local (12 GB VRAM-fitting) and Ollama Cloud frontier models. Eval and sweep code shouldn't care whether a model runs on the local GPU or on Ollama's cloud GPUs — those concerns belong to infrastructure. We also want: rate-limit handling, automatic retries on 429, fallback chains, spend tracking, OpenAI-compatible API surface (so any client lib works).

## Decision

A single **LiteLLM proxy** (Podman container, port 4000, backed by Postgres for spend tracking) routes all model calls. The proxy's `api_base` for every model — local AND cloud — points at the **local Ollama daemon** at `http://host.containers.internal:11434`. The local daemon already proxies cloud models via the user's ed25519 signin, so we don't need to manage `OLLAMA_API_KEY` inside the LiteLLM container.

Every model gets a stable `model_name` (e.g. `qwen3-14b-q4`, `gpt-oss-120b-cloud`). Eval/sweep code references the model name only; backend routing is config.

LiteLLM applies a `max_tokens=16384` clamp on cloud models (matches Ollama Cloud's hard output cap). Fallback chains: `qwen3-coder-480b-cloud → gpt-oss-120b-cloud → gpt-oss-20b-cloud → qwen3-14b-q4`.

## Consequences

- **Easier**: one OpenAI-compatible endpoint for everything; swap-in/out of models is a config edit, not a code change; spend tracking is automatic; the local daemon's cloud signin is the sole place that knows the cloud credentials.
- **Harder**: a second daemon to keep running (Ollama daemon is the actual gateway; LiteLLM is the router). Failures in either need diagnosis.
- **Risks**: if the local Ollama signin expires, all cloud calls 401. If LiteLLM's spend DB grows large, may need pruning.

## Considered alternatives

- **Direct Ollama Python client per call** — fine for prototyping; loses LiteLLM's retry/fallback/spend benefits, and forces every script to handle cloud-vs-local routing.
- **`OLLAMA_API_KEY` injected into the LiteLLM container** — works but adds a secret to manage; the local-daemon-as-gateway pattern is simpler and equally functional.
- **Separate Ollama Cloud SDK** — Ollama Cloud uses the same API as local Ollama; no separate SDK needed.
