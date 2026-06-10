# I benchmarked local coding agents on a 12GB GPU — a 12B model beat a 70B, and the reason is tool-calling

*Jun 2026 — single-machine local-only benchmark, no API calls*

---

I've been building a local agentic eval harness on my homelab server's RTX 3080 Ti (12GB VRAM) and wanted to share what I found. The headline result is counterintuitive: **Gemma 4 12B and Qwen3-Coder 30B both hit 100% pass rate; Llama 3.3 70B hit 58%; Qwen2.5-Coder 32B hit 0%.** Coding benchmark leaderboard rankings don't predict this at all, and the reason is straightforward once you look at the trajectories.

---

## Setup

All models run locally via ollama and llama-swap behind a litellm proxy — no external API calls. The harness is a ReAct scaffold with six sandboxed tools: `fs_read`, `fs_write`, `fs_grep`, `shell_exec`, `python_eval`, `http_fetch`. Scoring is done with inspect-ai; runs are tracked in MLflow with OpenTelemetry traces (postgres backend, 10k+ scored runs accumulated). Each model gets temp=0, 3 seeds, across 12 tasks in **pbs-agent-v0.1**: file operations, shell pipelines, HTTP + compute chains, and code-fix tasks. That's 36 scored runs per model.

Models tested:

| Model | VRAM footprint | Pass rate (36 runs) | Mean time/cell |
|---|---|---|---|
| gemma4-12b | fully on-GPU | **100%** | 13.9s |
| Qwen3-Coder-30B (MoE, 3B active) | fully on-GPU | **100%** | 19.2s |
| Qwen2.5-Coder-32B | fully on-GPU | **0%** | 19.2s |
| Llama-3.3-70B Q4 | hybrid CPU/GPU offload | **58.3%** | ~90s/cell |

---

## What actually fails

I was expecting quality differences — wrong tool choices, bad code, incomplete plans. What I actually found by reading trajectories was much more boring and more damning: **the failing models emit tool calls as plain text instead of structured `tool_calls`.**

Qwen2.5-Coder-32B on a task requiring an HTTP fetch followed by a computation:

```
{"name": "http_fetch", "arguments": {"url": "http://fixture/data.json"}}
{"name": "python_eval", "arguments": {"code": "result = data['value'] * 2"}}
```

That's not a tool call. That's text that looks like tool calls. The harness sees an assistant message with content and no `tool_calls` field, correctly treats it as a final answer, and the task fails because nothing was executed.

Llama-3.3-70B on a grep task:

```json
{"type": "function", "name": "fs_grep", "parameters": {"path": "/workspace", "pattern": "TODO"}}
```

Again, content text. Zero structured tool calls across the whole trajectory. On another task it skipped the tool entirely and wrote:

```python
import hashlib
data = open('/workspace/file.txt').read()
print(hashlib.sha256(data.encode()).hexdigest())
```

Raw Python in the message body, as if the conversation were a REPL.

The pattern is consistent: the first tool call often works correctly (model gets lucky with the initial turn), but as chains lengthen past 1–2 steps, both models fall back to narrating what they would do rather than invoking tools.

---

## The 70B in more detail

Llama 3.3 70B Q4 runs in hybrid mode — 10 GPU layers (had to tune down from the default 16; `cudaMalloc` OOM on the compute buffer caused llama-server segfaults at higher offload). That costs roughly 6× latency vs. the 12B (~90s/cell vs. ~14s).

I tried the obvious fix: a dedicated Llama-3.3 chat template (`--chat-template-file`) with full tool-call handlers. It didn't help. The model still emits text-format calls at the same rate.

The failure pattern is also not uniform across tasks. The 70B passed all 7 tasks that needed short chains — it even completed a 4-tool sequence correctly. But it went 0/3-seeds on the 5 tasks requiring it to **continue after receiving tool output**. It stops after fetching or executing once, then writes a summary paragraph instead of issuing the next call. This looks like a training data problem, not a context-length or quantization problem.

---

## The fallback experiment

I added a harness fallback: parse content-embedded JSON that looks like tool calls and execute them anyway. This worked mechanically — Qwen2.5-Coder went from 1 executed tool call to 6 on the test task. But it did not rescue scores.

The recovered calls contained pseudo-code like:

```python
content = $response['content']
result = process(content)
```

The model is assuming a variable-binding environment that doesn't exist in a stateless tool-call loop. It's not formatting wrong — it's reasoning in a different execution model entirely. The harness shim can fix the envelope; it can't fix the agent's mental model of how state flows between steps.

**Conclusion from the fallback:** the weakness is in agentic training, not output format. Patching the format just uncovers the next layer of failure.

---

## Why the small models win

Gemma 4 12B and Qwen3-Coder 30B (MoE, ~3B active params) both emit clean structured `tool_calls` on every turn, follow tool outputs correctly, and chain multi-step tasks without degradation. Neither is doing anything special — they're just complying with the protocol the harness expects.

Qwen3-Coder is explicitly marketed as agentic-training-focused. Gemma 4's recent release likely included similar data. Qwen2.5-Coder 32B is an excellent code-completion model — it scores well on HumanEval-style benchmarks — but it was not trained for this kind of tool-use loop.

The lesson I keep coming back to: **agentic capability is not a function of model size, and it's not the same thing as coding ability.** A model that can write flawless Python from a docstring is not necessarily a model that can operate a tool loop. These are different behaviors that require different training data.

---

## The hard suite: the tie breaks, and a third failure mode appears

The easy suite saturated (two models at 100%), so I built a 32-task hard suite — code, data, shell, and multi-hop categories: multi-file bug hunts, ETL pipelines over multiple files, multi-hop HTTP fixture chains, log-analysis pipelines with deliberate edge cases. Every task has a machine-verified answer. I also added Devstral Small 24B, Mistral's agentic-coding model.

Results (32 tasks, react scaffold, temp 0, single seed — n=32 cells per model):

| Model | Hard-suite pass rate |
|---|---|
| Gemma 4 12B | **93.8%** |
| Qwen3-Coder-30B | 78.1% |
| Devstral Small 24B | 37.5% |

The 12B model won again — on the *hard* suite, against models 2–2.5× its size, including one purpose-built for agentic coding.

Devstral's 37.5% turned out to be its own story. Its trajectories showed a third failure mode, distinct from the malformed-JSON problem: **it narrates instead of acting.** On long multi-step tasks it replies with a friendly plan — "I'll help you implement Kahn's algorithm... let's break this down into steps" — and markdown *pseudo-code* showing the tool calls it would make (```ops = fs_read('src/ops.txt')```), but makes zero actual tool calls. The episode ends on the spot.

An exact-replay A/B isolated the trigger: with the harness's polite system prompt ("You are an assistant with tool access...") Devstral narrates; append one sentence — *"act only via tool calls; never describe or plan in text; a reply without a tool call ends the session"* — and the same model on the same task immediately emits a proper structured call.

So I re-ran the whole suite with that sentence added for every model:

| Model | v1 prompt | v2 (act-don't-narrate) | Δ |
|---|---|---|---|
| Gemma 4 12B | 93.8% | 93.8% | 0 |
| Qwen3-Coder-30B | 78.1% | 81.3% | +3.2 |
| Devstral Small 24B | 37.5% | 53.1% | **+15.6** |

One sentence in the system prompt was worth 15+ points to one model, ~3 to another, and nothing to the leader. That's the second lesson of this whole exercise: **prompt robustness is a model property, and benchmarks silently measure it.** Gemma 4 acts correctly under a generic prompt; Devstral needs to be told, firmly, in language resembling its training scaffold. If you benchmark with a single shared prompt — which is the fair default — you're partly measuring how much each model's agentic training generalizes beyond its home scaffold.

Even with its fair shot, Devstral's 53% against Gemma 4's 94% says the rest of the gap is real capability on these tasks, concentrated in long multi-hop chains (its multi-category score: 38%).

---

## Appendix: harness details

**Infrastructure:** Arch Linux, RTX 3080 Ti (12GB), ollama + llama-swap for model management, litellm proxy for a uniform OpenAI-compatible endpoint. All local, no internet calls during eval.

**Harness:** Custom ReAct scaffold (bridged through inspect-ai). Each task is one episode: system prompt + task description, then a loop of `{assistant_message → tool_dispatch → tool_result}` until the model replies without a tool call or hits the turn/tool budget. Tools run in Podman + gVisor sandboxes — each episode gets a fresh containerized workspace, with tools served over MCP from a pooled set of servers; HTTP tasks use offline fixtures behind a host allowlist, so nothing touches the network.

**Scoring:** deterministic end-state predicates over the final workspace — file-contains/equals/exists checks, DB queries, and composite `all_of` predicates. Every task's expected answer was computed and verified before the predicate was written. Pass rate = mean score across seeds.

**Observability:** MLflow experiment tracking (one run per model per task per seed) with MLflow Tracing + OpenTelemetry spans (Tempo), and raw trajectory JSONL archived to S3 (MinIO). Trajectory inspection is how I found the malformed tool-call patterns — the full message sequences are queryable.

**Reproducibility:** The postgres schema, MLflow instance, and harness code all live locally; nothing is behind a paid API. The only external dependency is the model weights themselves (all publicly available on Hugging Face/Ollama hub).
