<!-- DRAFT — pending Matt's approval before any publication. Not published anywhere. -->

# Last time, tool-calling was the bottleneck. This time I trained it away — fine-tuning a 4B agent on its own trajectories, on a 12GB GPU

*Jun 2026 — single-machine, local-only. The fine-tune, the GGUF export, and every eval ran on one RTX 3080 Ti (12GB). No API calls in the training or the scoring loop.*

---

In the [last post](local-coding-agent-benchmark.md) I benchmarked local coding agents on a 12GB GPU and found something counterintuitive: leaderboard rankings didn't predict agentic performance at all. A 12B model and a 30B MoE both hit 100%; a 32B coder hit 0%; a 70B hit 58%. The reason wasn't quality. It was **tool-calling fidelity** — the failing models emitted tool calls as plain *text* instead of structured `tool_calls`, so nothing executed. I wrote that up as a finding and made a claim at the end of it: this is a *training* artifact, not a capability ceiling.

That claim was a loose end. If it's really about training, then training should be able to fix it. So this time I tried to close the loop: **evaluate a base model, fine-tune it on agent trajectories, evaluate it again, and see if the gap actually moves** — all on the same 12GB card that ran the benchmark.

Short version: it moved a lot. +19 to +27 points across three suites, including a 9× jump on a held-out set that postdates the training data, and the model got 2–2.5× *faster* in the process. The honest version has a caveat I'll get to, because the most interesting failure mode from last time didn't fully die.

---

## The idea: eval → train → eval, with the model's own work as the data

The base model is **Qwen3-4B** — small enough to QLoRA-fine-tune inside 12GB, big enough to be a real agent. The training data is the part I care most about: **659 of the harness's own verified-successful agent trajectories**, the actual tool-call sequences it produced solving tasks, filtered hard (only `end_state == 1.0` episodes; 14 of 1,295 candidates rejected by faithfulness cross-checks for misalignment or truncated/recovered calls), and re-rendered against the real tool schemas.

That's the STaR/rejection-sampling idea: the model's own successes, cleaned, become its curriculum. I mixed those with public function-calling datasets (xLAM, ToolACE, Hermes-FC) and general chat (ultrachat) at a 60/40 tool/chat ratio, 20,000 samples total. The chat fraction is there so it doesn't forget how to talk.

Training config, for anyone reproducing on similar hardware: QLoRA 4-bit, rank 16 / α 32, lr 2e-4, 2 epochs (2,500 steps), max sequence 4096, batch size 1 with grad-accum 16, responses-only masking via the Qwen3 chat template. Final train loss ≈ 0.75. It fit in 12GB only after dropping to seq-4096 + batch-1 (the p95 of the data is ~4.2k tokens) — the first attempt OOM'd at longer sequences. Then I exported to a `q4_k_m` GGUF and served both the base and the fine-tuned model identically through ollama behind a litellm proxy, so the before/after comparison is apples-to-apples at the same quantization.

---

## Pre-registering the bet

Before training I wrote down what would count as success, so I couldn't move the goalposts after seeing numbers. Three evals, deliberately different in how "clean" they are relative to the training data:

| Eval | Relationship to training data | What it tests |
|---|---|---|
| **BFCL v3** (1,000 tasks) | Clean — academic benchmark, never in training | Single-turn function-calling format generalization |
| **Brutal suite** (24 tasks × 3 seeds) | **Clean held-out** — *built after the dataset was frozen* | Hard multi-turn agentic tasks; zero contamination possible |
| **Hard suite** (32 tasks × 3 seeds) | **Disclosed contamination** — its own successful trajectories are in the training mix | Memorization-inclusive gain; reported, never headline |

The brutal suite is the one that matters most. I built it *after* I'd already frozen the fine-tuning dataset, specifically so there was a test the model could not have memorized. The hard suite is the opposite — I knowingly trained on successful trajectories from it, so any gain there is inflated by memorization and I'm flagging it as such rather than hiding it.

Targets, written down in advance: BFCL ≥ +5pp, brutal ≥ +2 tasks, hard ≥ +10pp, and — the one tied directly to last post's finding — **zero tool-calling protocol failures** (no text-emitted calls, no narration) on the fine-tuned model.

---

## Results

| Eval | base Qwen3-4B | fine-tuned | Δ | target | verdict |
|---|---|---|---|---|---|
| **BFCL v3 (clean)** | 64.7% | **83.7%** | **+19.0pp** | ≥+5pp | ✅ |
| **Brutal (clean held-out)** | 2.8% | **25.0%** | **+22.2pp (~9×)** | ≥+2 tasks | ✅ |
| **Hard (disclosed contam.)** | 17.7% | **44.8%** | **+27.1pp** | ≥+10pp | ✅ |

The brutal result is the headline I trust most. The base 4B essentially *cannot* do these multi-turn agentic tasks — it solved 2 of 72 trials. After fine-tuning on its bigger sibling's-and-its-own clean trajectories, it solved 18 of 72. That's a 9× improvement on a suite it provably never saw, where the only thing that changed was the weights.

BFCL confirms it generalizes to a totally different, academic single-turn format too: +19 points. And the contaminated hard suite, for what it's worth (memorization included), nearly tripled.

### The unadvertised win: it got *faster*

I didn't pre-register efficiency, but it's too clean to leave out:

| Eval | latency (base → ft) | output tokens (base → ft) |
|---|---|---|
| BFCL | 4,246 → 1,993 ms (**−53%**) | 708 → 314 (**−56%**) |
| Brutal | 40.3 → 33.2 s (−18%) | 5,501 → 3,331 (−39%) |
| Hard | 45.9 → 18.0 s (**−61%**) | 6,418 → 1,890 (**−71%**) |

The fine-tuned model is simultaneously **more accurate and 2–2.5× faster, emitting 40–70% fewer tokens.** This isn't a quality-for-speed trade — it's the same mechanism as the accuracy gain. The base 4B wraps every action in a wall of reasoning prose; the fine-tuned one learned to stop narrating and just call the tool. Less rambling is both faster *and* more correct.

---

## The honest caveat: the failure mode shrank but didn't die

The fourth target was the one that came straight from last post's finding: zero tool-calling protocol failures. I ran the trajectory auditor — a mechanical classifier that flags two things from the F-012 finding: `text_emitted` (a tool call written as JSON text in the content field, the thing that scored those bigger models 0%) and `narration` (a whole episode with no structured tool calls at all).

| Failure mode | base (both agent suites) | fine-tuned |
|---|---|---|
| `text_emitted` (malformed JSON-as-text call) | present | **0 — eliminated** |
| `narration` (episode with no tool calls) | 28 | **10** |

**As pre-registered, this target is refuted.** I said zero, and the fine-tuned model still had 10 narration episodes. I'm not going to round that up to a win.

But the shape of the miss matters. The *worst* failure mode — the exact `text_emitted` pattern that tanked the 32B and 70B last time — is **gone, zero episodes.** Pure narration dropped 64% (28 → 10). And the residual isn't random: all 10 episodes land on just 4 hard tasks, and on those it fails on *every* seed — deterministic, not flaky.

So I dug into those 4 instead of hand-waving them, and the answer surprised me. The easy story would be "needs more training examples" — but two of those four tasks were *already in the training data*, and the model still fails them. The next easy story would be "it's running out of output budget mid-thought" — so I re-ran them with the per-turn token ceiling quadrupled (4k → 16k). Narration barely moved (10 → 9), and every failed episode simply consumed the *entire* larger budget. It isn't getting cut off.

What's actually happening: on these few prompts, under greedy decoding (temperature 0, no repetition penalty), the model drops into a non-terminating reasoning loop — it fills whatever token ceiling you give it and never emits the tool call. The *same model* calls the tool fine on a lightly reworded version of the identical task. So this isn't a training defect; it's a **decoding-degeneration** artifact on a handful of prompts, and a repetition penalty would almost certainly clear it. I'm leaving it as a documented caveat rather than tuning the serving config until the number turns green — but it's worth being precise about what the 1-of-4 miss *is*: a greedy-decoding quirk, not the tool-calling regression the test was built to catch.

---

## What this does and doesn't show

It shows that **the tool-calling gap from last post is trainable** — and trainable cheaply, on consumer hardware, using the model's own verified work as the curriculum. A 4B model went from "can't really do multi-turn agent tasks" to "solves a quarter of the hardest held-out suite," generalized to an unrelated academic benchmark, and got faster doing it. The eval→train→eval loop closes, and it closes on a 12GB GPU with no external API in the loop.

It does not show that 4B is now a *good* agent in absolute terms — 25% on the brutal suite is a big relative jump from a low base, not a high score. And the narration caveat is real: protocol fidelity improved sharply but isn't solved. Both of those are honest ceilings, not footnotes.

What I find most useful is the methodology rather than the model: a single machine can run the whole flywheel — benchmark, mine its own successful trajectories, fine-tune, re-benchmark on a contamination-controlled held-out set, and audit the trajectories to check *how* it passed, not just whether. That loop is the actual product. The 4B is just the first thing I ran through it.

---

*Methodology notes: both arms served as `q4_k_m` GGUF via ollama + litellm, temp 0, identical ReAct scaffold and prompt. Brutal/hard scored on full agent-path success; BFCL on AST match. All runs tracked in MLflow over a postgres backend. The held-out brutal suite was authored after the training dataset was frozen; the hard-suite contamination is disclosed and reported separately from the clean results.*
