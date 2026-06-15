---
doc_id: f-009-qwen3-30b-moe-refuted-h1-invalid
title: 'F-009: EXP-006 — H1 replication REFUTED at 0.583 vs anchor 0.750; sweep
  INVALID. Per pre-reg, H2/H3/H4 verdicts not load-bearing. qwen3-30b-a3b-moe
  NOT promoted; root-cause filed.'
zone: lab
kind: finding
status: active
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
depends_on:
- kind: doc
  target: exp-006
- kind: doc
  target: f-005-12gb-agent-v0-2-tool-use
- kind: code
  target: lab:scripts/analyze_exp006.py
- kind: artifact
  target: lab:analysis/EXP-006/SUMMARY.md
- kind: artifact
  target: lab:analysis/EXP-006/verdicts.md
- kind: artifact
  target: lab:analysis/EXP-006/per_task_endstate.csv
- kind: artifact
  target: lab:analysis/EXP-006/per_cell.csv
- kind: artifact
  target: lab:analysis/EXP-006/gap_closure.csv
tags:
- lab
- finding
- findings
- moe
- qwen3
- 30b
- agent
- replication-failure
- confidence-high
- importance-7
---

# F-009: EXP-006 — H1 replication REFUTED; sweep INVALID; qwen3-30b-a3b-moe NOT promoted

## TL;DR

EXP-006 ran the full 288-cell sweep (12 tasks × 3 models × 8 seeds) at 288/288
done, 0 errors — clean operationally. The pre-registered decision rules then
fire as written:

- **H1 — Replication. REFUTED — sweep INVALID.** qwen3-14b-q4 (reasoning-OFF)
  end_state observed = **0.583** [0.479, 0.688] (n=96, bootstrap CI), vs F-005
  anchor 0.750 ± 0.05. `|observed − anchor| = 0.167`, ~3.3× the pre-reg band.
  Per the pre-reg, H2/H3/H4 verdicts are recorded but **not load-bearing**:
  decisions taken from them would be unsupported.
- **H2 — Headline lift. REFUTED (not load-bearing).** qwen3-30b-a3b-moe
  end_state = **0.583** [0.490, 0.677] (n=96), threshold ≥ 0.850. The MoE
  arm tied the dense baseline on end_state.
- **H3 — Gap closure. REFUTED (not load-bearing).** `gap_closure = 0.000`
  (dense=0.583, moe=0.583, cloud=0.969); MoE closed 0 % of the local-vs-cloud
  end_state gap on this sweep.
- **H4 — Tool-correctness ceiling. REFUTED (not load-bearing).** MoE
  tool_correctness = **0.500** [0.406, 0.594] (n=96), threshold ≥ 0.95. The
  MoE arm fires zero tool calls on 40/96 cells (5/12 tasks: both http, both
  shell, multi-db-self-check) — the F-005 "llama3.1 narrate-instead-of-call"
  pattern, now in qwen3-30b-a3b-moe.

**Decision per pre-reg "H1 fail" path**: qwen3-30b-a3b-moe is **NOT promoted**
as the lab's default local model. The dense `qwen3-14b-q4` (reasoning-OFF)
stays the local default. The H1 failure is rooted in legitimate lab plumbing
changes between EXP-002 (May 2026) and EXP-006 (this sweep) — F-005's
"Surprises" 1 and 2 have been partially fixed in the intervening tasks/scorers
work, which moves the anchor; the F-005 baseline of 0.750 is no longer the
right one to replicate against. EXP-006b is filed to re-run with a recomputed
H1 anchor against the current task/scorer revision; no claim about the MoE
arm is made until that lands.

The MoE arm's H4 result (tool_correctness 0.500 with 40 zero-tool-call cells)
is **independently** a strong signal that qwen3-30b-a3b-moe via llama-swap +
llama.cpp's `--jinja` tool-call template has a tool-wiring weakness vs ollama's
qwen3-14b-q4. This stands regardless of H1 — even if H1 had replicated, H4
would have refuted at 0.500 vs the 0.95 threshold.

## Setup

- **Experiment:** EXP-006 (plan:
  [`docs/exp/EXP-006-qwen3-30b-moe-vs-14b-dense.md`](../exp/EXP-006-qwen3-30b-moe-vs-14b-dense.md),
  pre-reg commit `dbb2f8a`; pre-sweep plumbing fix commit `dbfcbd5`)
- **Sweep config:**
  [`conf/sweep/EXP-006.yaml`](../../conf/sweep/EXP-006.yaml)
- **Models (3):**
  - `qwen3-14b-q4` (local ollama, `think:false` + `keep_alive:0`) —
    replication arm (H1)
  - `qwen3-30b-a3b-moe` (local llama.cpp Phase 19d CUDA build via llama-swap,
    `-ngl 99 -ot exps=CPU`, ~5.6 GB peak VRAM) — treatment arm (H2, H3, H4)
  - `gpt-oss-120b-cloud` (Ollama Cloud Pro) — ceiling reference (H3)
- **Tasks (12):** PBS-Agent v0.1, identical task list to EXP-002.
- **Config (1):** `greedy-1024` — `temperature=0.0`, `top_p=1.0`,
  `max_tokens=1024`. Identical to EXP-002 / F-005.
- **Cells:** 12 tasks × 3 models × 1 config × 8 seeds = **288 runs.**
- **Pass rate:** **288/288 done, 0 errors.** No kill criterion fired.
- **Sandbox image hash:** `797b129dd385e712ed232edf424b00fabd329c0e3395fb945478bde1ab4a8739`,
  single hash across all 288 cells (no drift; F-005 Surprise 4 doesn't apply).
- **Spot-check:** trajectories pulled per-cell from DB via the standard
  analyzer; per-cell `score_breakdown` values match the analyzer's aggregates.

## Per-hypothesis verdict

### H1 — Replication of F-005's qwen3-14b-q4 baseline · REFUTED — sweep INVALID

Pre-reg: `|mean(end_state | qwen3-14b-q4, all 96 cells) − 0.750| ≤ 0.05`.

| qwen3-14b-q4 | n | mean end_state | 95 % bootstrap CI |
|---|---|---|---|
| EXP-006 (this sweep) | 96 | **0.583** | [0.479, 0.688] |
| F-005 (EXP-002, anchor) | 96 | 0.750 | F-005 § H2 |

`|0.583 − 0.750| = 0.167 ≫ 0.05`. **REFUTED — sweep INVALID** per pre-reg.

Per-task comparison vs F-005 (qwen3-14b-q4 only):

| task | F-005 end_state | EXP-006 end_state | Δ | notes |
|---|---|---|---|---|
| code-find-and-fix-bug | 1.000 | **0.000** | −1.000 | F-005 Surprise 2: predicate trivially passable. Predicate has been tightened since; what was a "free pass" in F-005 is now a real fail for qwen3-14b-q4. |
| code-read-and-explain | 1.000 | 1.000 | 0.000 | stable |
| code-write-and-execute | 1.000 | 1.000 | 0.000 | stable |
| fs-grep-extract-and-write | 1.000 | 1.000 | 0.000 | stable |
| fs-read-and-copy | 1.000 | 1.000 | 0.000 | stable |
| fs-write-csv-summary | 1.000 | 1.000 | 0.000 | stable |
| http-fetch-and-count | 0.000 | 0.000 | 0.000 | F-005 had fixture broken; now fixture serves but qwen3-14b-q4 fails downstream extraction (uptime.txt did not contain '4242'). tool_correctness is still 1.0 — model calls http_fetch correctly. |
| http-fetch-and-extract | 0.000 | 0.000 | 0.000 | same shape as above. |
| multi-db-self-check | 1.000 | 1.000 | 0.000 | stable |
| multi-words-and-hash | 1.000 | **0.000** | −1.000 | NEW failure. tool_correctness still 1.0 (python_eval called). The hash check downstream is now failing for this model (hash.txt did not contain expected SHA). Task or scorer drift since EXP-002 — worth a follow-on. |
| shell-count-lines | 0.000 | 0.000 | 0.000 | stable (still zero) |
| shell-pipeline-extract | 0.000 | **1.000** | +1.000 | NEW improvement. Plausibly tool-surface drift in the shell scaffold. |

The shifts net to −0.167 vs the F-005 anchor. Two are downward (predicate
tightening + multi-words regression), one is upward (shell-pipeline-extract).
**The lab plumbing has drifted in ways that are visible — and visible in both
directions.** That's healthy from a measurement-validity standpoint and exactly
why the H1 replication guard exists, but it means the F-005 anchor of 0.750
is no longer the right number to replicate against.

Excluding only the http category (which F-005 flagged as fixture-broken),
the dense end_state is **0.700**. Excluding both http and
`code-find-and-fix-bug` (the trivially-passable F-005 Surprise 2 predicate),
the dense end_state is **0.778** — within the ±0.05 band of 0.750 on the
remaining 9/12 tasks. So the H1 anchor of 0.750 was, in retrospect,
load-bearing on (a) the broken http fixture and (b) the trivially-passable
code-bug predicate — fix either and the anchor moves.

### H2 — Headline lift (qwen3-30b-a3b-moe end_state ≥ 0.850) · REFUTED (not load-bearing)

| qwen3-30b-a3b-moe | n | mean end_state | 95 % bootstrap CI |
|---|---|---|---|
| EXP-006 | 96 | **0.583** | [0.490, 0.677] |

Threshold: ≥ 0.850. Observed CI upper bound is 0.677 — even the optimistic
end of the CI is 0.173 below the threshold. **REFUTED** by a comfortable
margin. Per pre-reg, this verdict is not load-bearing because H1 invalidated
the sweep.

Per-task: MoE matches dense exactly on the 6 fs/code tasks where dense
already wins (perfect on both), drops zero on both http and both shell
tasks (same shape as dense for http; **worse** than dense on
shell-pipeline-extract, where dense got 1.0 and MoE got 0.0). MoE picks up
a win over dense on `multi-words-and-hash` (1.0 vs dense's 0.0), but loses
exactly that win on `shell-pipeline-extract` (0.0 vs dense's 1.0).
**Per-task, the MoE arm is not uniformly better or uniformly worse — it's
trading tasks with dense.** That's not the headline behaviour the H2 pre-reg
expected from a larger, putatively-stronger planner.

### H3 — Gap closure (≥ 0.50) · REFUTED (not load-bearing)

Pre-reg: `gap_closure := (moe − dense) / (cloud − dense) ≥ 0.50`,
all on the same 96-cell denominator (12 tasks × 8 seeds).

| term | value |
|---|---|
| dense (qwen3-14b-q4) end_state | 0.5833 |
| moe (qwen3-30b-a3b-moe) end_state | 0.5833 |
| cloud (gpt-oss-120b-cloud) end_state | 0.9688 |
| denom (cloud − dense) | +0.3854 |
| numer (moe − dense) | 0.0000 |
| **gap_closure** | **0.000** |

Per pre-reg, the denominator is well-defined (cloud − dense > 0) so H3 is
not UNDEFINED. **gap_closure = 0.000 < 0.50: REFUTED**, again per pre-reg
not load-bearing because H1 invalidated the sweep.

Per-category gap closure (also in
[`analysis/EXP-006/gap_closure.csv`](../../analysis/EXP-006/gap_closure.csv)):

| category | dense | moe | cloud | gap_closure |
|---|---|---|---|---|
| code | 0.667 | 0.667 | 1.000 | 0.000 |
| fs | 1.000 | 1.000 | 1.000 | UNDEFINED (denom = 0; all 3 at ceiling) |
| http | 0.000 | 0.000 | 1.000 | 0.000 (MoE closed nothing — and unlike dense, didn't even call the tool) |
| multi | 0.500 | 1.000 | 0.812 | 1.600 (over-closes — MoE > cloud here, driven by multi-words-and-hash) |
| shell | 0.500 | 0.000 | 1.000 | **−1.000** (MoE regresses vs dense; dense's shell-pipeline-extract win is lost) |

The category-level pattern is: MoE is **not differentially closing the gap**;
it's swapping wins with dense at category granularity. Cloud is at near-ceiling
(0.969) on every category except multi-words-and-hash.

### H4 — Tool-correctness ceiling (qwen3-30b-a3b-moe ≥ 0.95) · REFUTED (not load-bearing, but independently meaningful)

| qwen3-30b-a3b-moe | n | mean tool_correctness | 95 % bootstrap CI |
|---|---|---|---|
| EXP-006 | 96 | **0.500** | [0.406, 0.594] |

Threshold: ≥ 0.95. CI upper bound 0.594 — 0.356 below the threshold.
**REFUTED**, and **decisively** so.

Mechanism: **40 of 96 MoE cells (42 %) fired zero tool calls.** Distribution:

| task | n cells with zero tool calls | n total | MoE end_state | MoE tool_correctness |
|---|---|---|---|---|
| http-fetch-and-count | 8 | 8 | 0.000 | 0.000 |
| http-fetch-and-extract | 8 | 8 | 0.000 | 0.000 |
| multi-db-self-check | 8 | 8 | 1.000 | 0.000 |
| shell-count-lines | 8 | 8 | 0.000 | 0.000 |
| shell-pipeline-extract | 8 | 8 | 0.000 | 0.000 |

This is the **F-005 "llama3.1 narrate-instead-of-call" pattern**, now seen
in qwen3-30b-a3b-moe via the llama-swap + llama.cpp `--jinja` path. The
MoE model decides not to invoke any tool on these 5/12 tasks and terminates
at turn 1. Comparing tool_call_count distributions:

| model | mean turns | p50 | p95 | mean tool calls | zero-tool cells |
|---|---|---|---|---|---|
| gpt-oss-120b-cloud | 3.39 | 3 | 5 | 2.39 | 0/96 |
| qwen3-14b-q4 (dense) | 3.00 | 3 | 6 | 2.57 | 0/96 |
| qwen3-30b-a3b-moe | 2.17 | 2.5 | 4 | 1.17 | **40/96** |

qwen3-30b-a3b-moe's median trajectory is half-a-turn shorter than dense's
and fires less than half the tool calls. This is **not** a planner-capacity
gap; it's a tool-emission gap consistent with the model template / chat
template / `--jinja` behaviour on the llama.cpp serve path.

**H4 stands as a real, independent finding even though H1 invalidated the
sweep.** A qwen3-30b-a3b-moe arm with tool_correctness 0.500 cannot be
trusted as a tool-use default regardless of what end_state it would have
gotten with full tool wiring; the model is not reliably emitting tool calls
on the lab's PBS-Agent v0.1 task surface.

The single off-pattern in the MoE zero-tool-call slice is
`multi-db-self-check`: MoE fires zero tool calls but **end_state = 1.0**.
Spot-check shows the task's success predicate is satisfied by sandbox-init
state (a `db_health` row pre-seeded by the task setup), so any model that
no-ops gets the point — this is a **second** "trivially-passable success
predicate" task, in the same shape as F-005 Surprise 2. Filed as follow-on:
tighten the `multi-db-self-check` predicate before the next agent sweep.

## Comparison vs F-005 (qwen3-14b-q4 anchor)

The pre-reg required this sanity-check table:

| metric | F-005 (qwen3-14b-q4) | EXP-006 (qwen3-14b-q4) | Δ | status |
|---|---|---|---|---|
| mean end_state (with-http) | 0.750 | **0.583** | −0.167 | outside ±0.05 band |
| mean end_state (without-http) | 1.000 (= 9/9 in F-005 § "Surprises") | 0.700 | −0.300 | not directly comparable; F-005's no-http baseline excluded 80 cells |
| mean tool_correctness | 1.000 | 0.990 (95/96; one cell on `code-find-and-fix-bug` was 0/8 at 0.875 mean) | −0.010 | stable; the dense model's tool-call wiring is unchanged. |
| mean budget_respected | 1.000 (from F-005's H4 budget headroom claim) | 1.000 | 0.000 | stable |
| zero-tool-call cells | 0/96 | 0/96 | 0 | stable |

The dense model's **tool_correctness has not regressed**. The end_state
regression is entirely concentrated in the three "explanation" tasks above
(predicate tightening, multi-words-and-hash regression, http-extract
downstream); the dense model's planning is not the load-bearing change.

## Trajectory-judge slice

The judge fires only on `code-read-and-explain`. 1 task × 3 models × 8 seeds
= **24 judge calls**. Judge model: `gpt-oss-120b-cloud`.

| model | mean_judge | nonzero | self-judge cells |
|---|---|---|---|
| gpt-oss-120b-cloud | 1.000 | 8/8 | 8/8 |
| qwen3-14b-q4 | 1.000 | 8/8 | 0 |
| qwen3-30b-a3b-moe | 1.000 | 8/8 | 0 |

All three models score 1.000 on the held-out judge slice. The judge agrees
with the deterministic `end_state` scorer for every cell on this task; no
new judge-disagreement pattern relative to F-005's single-seed disagreement
on llama3.1.

## Operational notes — MoE in production

This is the lab's first agent-sweep with a model served by llama.cpp
(Phase 19d CUDA build) routed via llama-swap (Phase 19b) and orchestrated by
model_pool (Phase 19c). Operational observations from the 288-cell run:

1. **No GPU lease contention, no llama-swap hangs, no daemon crashes.**
   Kill criteria did not fire (errored cells = 0/288, sandbox failures = 0,
   `gpu_lease_timeout` = 0, llama-swap `/running` always responsive).
   The Phase 19a/b/c/d stack ran 96 MoE cells back-to-back without
   incident.
2. **MoE per-cell latency is ~3.3× dense and ~7.5× cloud.**
   p50 latency:
   - cloud: **8.8 s**
   - dense (qwen3-14b-q4 via ollama): **20.2 s**
   - moe (qwen3-30b-a3b-moe via llama-swap+llama.cpp, `-ot exps=CPU`):
     **67.1 s**

   p95 latency tracks the same ratio (cloud 11.7 s / dense 49.5 s /
   moe 98.6 s). MoE's worst cell hit 100.6 s — well inside the 600 s
   per-cell timeout, but the wall-clock cost of `-ot exps=CPU` hybrid
   offload is real. Pre-reg called out this as a risk; it didn't blow
   the wall budget, but it's a 3-4× planning-budget multiplier vs dense.
3. **MoE fires fewer tool calls per cell — even on cells where it succeeds.**
   Mean tool calls/cell:
   - cloud: 2.39 (max 5+)
   - dense: 2.57
   - moe: **1.17**

   Of the 96 MoE cells, 40 (42 %) fired zero tool calls and 31 fired
   exactly 1. The MoE arm has a structurally shorter trajectory than the
   other two. This is consistent with the model deciding it's "done" early
   — either narrating an answer it never executed (the H4 mechanism above)
   or finishing in a single fs_write turn for simple tasks.
4. **VRAM headroom held.** Phase 19a smoke had clocked qwen3-30b-a3b-moe
   at ~5.6 GB peak GPU with `-ngl 99 -ot exps=CPU`. The plumbing fix in
   `dbfcbd5` (qwen3-14b-q4 `keep_alive=0`) freed ollama VRAM between arms
   as designed, and `model_pool.declare(plan)` (Phase 19c) handled the
   pre-flight + eviction without manual intervention. **No OOM, no
   sandbox failures, no mid-sweep llama-swap interventions.**
5. **Sandbox image hash drift: zero.** F-005 saw three distinct image
   hashes during its 2 h sweep; EXP-006 saw one hash across all 288 cells
   (same Containerfile lineage at `5c364cc`). The hash-drift guard
   tightening that F-005 flagged didn't need to fire; the post-F-005
   `podman image prune` schedule changes are holding.
6. **No token-count capture in this sweep.** `experiment_runs.tokens_in /
   tokens_out` are NULL for every cell — neither the ollama backend nor
   the llama-swap + llama.cpp backend wrote token counts via the LiteLLM
   path in this run. **Filed as follow-on**: wire up token capture for
   llama.cpp through LiteLLM (Phase 19e or its equivalent) so that future
   sweeps can report tokens in/out and cost-per-cell. This finding does
   not depend on token counts (cost weight ratios are not a decision rule
   in EXP-006).

## What this changes about the lab's local-first thesis

F-005 left the lab here: **"locals can call tools at near-cloud accuracy,
but they lose end_state because they can't chain tool calls into a working
multi-turn solution."** F-009 modifies that picture in three ways:

1. **The H1 anchor 0.750 was load-bearing on broken infrastructure** (broken
   http fixture and the trivially-passable code-bug predicate). With those
   fixed, the dense end_state on the 9 remaining tasks is 0.778 — the
   "local end_state is 0.75" claim only ever held in the presence of those
   specific bugs. The next EXP-006b should re-anchor at 0.700 (with-http
   over the current task set) or 0.778 (excluding tasks the lab is
   actively fixing predicates on).
2. **The MoE arm is not a planner upgrade on the lab's current PBS-Agent
   v0.1 task surface — it's a tool-wiring downgrade.** qwen3-30b-a3b-moe
   via llama.cpp + llama-swap fires zero tool calls on 5/12 tasks. The
   per-task pattern doesn't show a planner-capacity gain; it shows a
   tool-emission regression vs the ollama-served qwen3-14b-q4 dense
   model. The hypothesis that "the gap is a planner capacity gap" cannot
   be tested with the current MoE arm because the MoE arm doesn't issue
   the tool calls the planner is supposed to chain.
3. **Cloud (gpt-oss-120b-cloud) is now at 0.969 end_state on this task
   suite** — up from 0.833 in F-005. The 13.6 pp lift is mostly recoveries
   on the previously-broken http tasks (now fixture-served correctly for
   cloud) and shell-count-lines (where cloud now gets 1.0 vs F-005's
   1.0 already — that part's stable). The end_state ceiling for cloud
   has moved up; the local-vs-cloud gap on this task surface is
   correspondingly wider.

The decision landscape after EXP-006:

| Use case | Recommendation |
|---|---|
| Lab default local model | **Stay on `qwen3-14b-q4` (reasoning-OFF).** Dense is unchanged in tool-call quality and is materially better on this sweep than the MoE arm (no zero-tool-call cells; end_state tied at 0.583 but on tasks where the dense model actually fires tools). |
| MoE evaluation | **Investigate tool-emission template before any further sweep claims.** The 40-cell zero-tool-call slice is the load-bearing artifact; if it's a `--jinja` template / chat-template / tool-call-spec gap in the llama.cpp serve path, fix that first before re-running EXP-006. |
| Local-vs-cloud gap question | **Re-pose against a recomputed dense baseline** once the task/scorer fixes (http extraction failure mode, multi-words-and-hash regression, multi-db-self-check trivially-passable predicate) are landed. The 0.583-vs-0.969 gap in this sweep mixes "local model can't extract '4242' from a real http body" with "local MoE never called the tool"; those are two different stories. |

## Decision

**qwen3-30b-a3b-moe is NOT promoted as the lab default local model.**
The dense `qwen3-14b-q4` (reasoning-OFF, `keep_alive=5m` for normal use,
the EXP-006 sweep-local `keep_alive=0` was for VRAM-coexistence, not for
default routing) remains the lab default local model.

No changes to `conf/litellm-config.yaml` route priority. No ADR-008 is
written from this finding — promotion would have required H2 + H3 both
CONFIRMED (the strong-yes path per the task brief), neither is true.

## Recommended next steps

1. **Diagnose the MoE tool-emission gap.** Pull a trajectory from one of
   the 40 zero-tool-call cells (e.g. `qwen3-30b-a3b-moe / http-fetch-and-
   extract / seed=1`, `run_id=6da84fe093d0b2f95a99f8cb`) and inspect the
   model's raw output. If the model is emitting tool calls as content
   rather than tool-call JSON (the F-005 llama3.1 pattern), this is a
   chat-template / `--jinja` template fix in llama.cpp's serve path. If
   the model is just narrating without attempting tools, this is a
   prompt-engineering follow-on (few-shot tool exemplars in system) plus
   a possible reasoning-mode ablation (the EXP-006c follow-up the plan
   already pre-empted for if H2 passes — now also relevant if H2 fails).
2. **Re-anchor EXP-006b** against the current dense baseline. Run the
   dense arm only at N=8 against the present task/scorer revision to lock
   in the new anchor (expect ≈0.78 excluding the trivially-passable
   `multi-db-self-check`, ≈0.58 with the full current surface). Once
   the anchor is fresh, re-run the full 3-arm sweep with whatever MoE
   fix lands from step 1.
3. **Tighten `multi-db-self-check`'s success predicate.** Like F-005's
   `code-find-and-fix-bug`, this task is satisfiable by sandbox-init state
   — MoE no-ops and scores 1.0 on it. Same fix shape as the Surprise 2
   follow-on: invert / require a tool-action-produced state.
4. **Track multi-words-and-hash regression.** Dense qwen3-14b-q4 scored
   1.000 on this task in F-005 and 0.000 in EXP-006 (with
   tool_correctness still 1.000). Either the task changed or python_eval's
   sandbox state changed. This is the single biggest load-bearing shift
   in the H1 deficit (8 cells × 1pp = 8.3pp of the 16.7pp H1 deficit;
   `code-find-and-fix-bug` is the other 8.3pp).
5. **Token-count capture for llama.cpp.** Future agent sweeps should
   report tokens in/out and cost-per-cell; the LiteLLM passthrough for
   llama-swap-fronted backends is not currently populating these columns.

## Caveats and known limitations

1. **H1 REFUTED is the operational kill criterion.** Per pre-reg, H2/H3/H4
   verdicts are reported but not load-bearing. The MoE arm's verdicts
   (REFUTED on H2, H3, H4) are not used to argue against the MoE thesis
   in any setting other than "the MoE arm tied dense on end_state and
   was worse on tool_correctness on **this** sweep with **these** tasks
   and **this** llama.cpp serve path." A future EXP-006b with a refreshed
   H1 anchor and a fixed MoE tool-emission path could land entirely
   differently.
2. **No reasoning-mode ablation on MoE.** Per pre-reg, the MoE arm uses
   its stock default chat template. Reasoning-OFF on qwen3-30b-a3b-moe is
   the EXP-006c follow-on. F-008's reasoning-OFF result on dense
   qwen3-14b-q4 is not assumed to transfer.
3. **N=8 seeds, 12 tasks** is right for an agent sweep at this stage but
   thin for per-category significance. The "shell-pipeline-extract:
   MoE=0.000, dense=1.000" gap is 16 cells (8 each); the binomial CI on
   the 16-cell binary diff is wide and the verdict on that single task
   should not drive a model-replacement decision on its own.
4. **Greedy decoding only.** Same as EXP-002 / F-005. Same caveats.
5. **Single sandbox runtime.** Podman + gVisor runsc.
6. **Token counts missing for all 288 cells.** Wall-time latency is the
   only operational cost signal in this sweep.
7. **No paired permutation test.** The pre-reg called for a paired-by-task
   permutation test as a secondary signal. With the H1 sweep-INVALID
   verdict, the secondary signals are not load-bearing; the analyzer
   does not emit a permutation-test number in this run. If EXP-006b lands
   with H1 valid, the paired test should be added to the analyzer.

## Components NOT run end-to-end in EXP-006

- **EXP-006b** (refreshed H1 anchor with current tasks/scorers) — filed
  as follow-on, not run.
- **EXP-006c** (MoE reasoning-mode ablation) — out of scope per pre-reg.
- **EXP-006d** (MoE tool-emission template fix) — filed by this finding,
  not run.
- **70B quality-ceiling lane** (Phase 19e, EXP-006b in plan terms) — out
  of scope per pre-reg; Phase 19e is wiring it in parallel.
- **Token-count capture for llama-swap-fronted models** — filed as
  follow-on, not landed.
- **Permutation-test secondary signal** — pre-registered, deferred to
  the refreshed-anchor follow-up.

## Reproduction

```bash
cd /data/lab/code

# Confirm pre-registration
uv run lab exp show EXP-006

# Sweep (~2 hr wall, 288 cells; MoE arm ~67 s/cell p50)
uv run lab sweep run conf/sweep/EXP-006.yaml --enforce-pre-registration

# Deterministic + judge evaluators (judge runs only on include_judge tasks)
uv run lab eval apply EXP-006

# Verdicts + analysis CSVs
uv run python scripts/analyze_exp006.py
```

## Files

- Plan: [`docs/exp/EXP-006-qwen3-30b-moe-vs-14b-dense.md`](../exp/EXP-006-qwen3-30b-moe-vs-14b-dense.md)
- Sweep config: [`conf/sweep/EXP-006.yaml`](../../conf/sweep/EXP-006.yaml)
- Analyzer: [`scripts/analyze_exp006.py`](../../scripts/analyze_exp006.py)
- Analysis: [`analysis/EXP-006/SUMMARY.md`](../../analysis/EXP-006/SUMMARY.md)
- Verdicts detail: [`analysis/EXP-006/verdicts.md`](../../analysis/EXP-006/verdicts.md)
- Per-task CSV: [`analysis/EXP-006/per_task_endstate.csv`](../../analysis/EXP-006/per_task_endstate.csv)
- Per-cell CSV: [`analysis/EXP-006/per_cell.csv`](../../analysis/EXP-006/per_cell.csv)
- Gap-closure CSV: [`analysis/EXP-006/gap_closure.csv`](../../analysis/EXP-006/gap_closure.csv)
- Parent findings: [F-005](F-005-12gb-agent-v0.2-tool-use.md)
- Follow-ons filed: EXP-006b (refreshed H1 anchor), EXP-006c (MoE reasoning
  ablation, conditional), EXP-006d (MoE tool-emission template fix).
trust_level: reliability_confirmed

## Promotion history
- 2026-06-14: unverified -> verified (by Matt Rachlin)
- 2026-06-14: verified -> reliability_confirmed (by Matt Rachlin)
