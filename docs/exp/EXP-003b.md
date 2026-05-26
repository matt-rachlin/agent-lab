# EXP-003b: RAG-augmented agent — does kb_query change task success?

Date created: 2026-05-26
Status: planned
Pre-registered: (commit SHA filled by `lab exp register` at registration time)

## Question

Does giving a 12 GB local model access to a sealed knowledge base via the
`kb_query` MCP tool change its task success rate on KB-grounded tasks?
We pair the same task suite (`pbs-agent-rag-v0.1`, 6 tasks) against the
same 5 models from EXP-002, run each cell **with** kb_query in the
tools list and **without** (forced to answer from training knowledge
alone), and compare end-state and tool-call scorers across the two
conditions.

EXP-003a measured retrieval quality in isolation. EXP-003b measures
whether the retrieval *helps the agent*.

## Hypothesis

Four pre-registered hypotheses on `pbs-agent-rag-v0.1` (6 tasks),
greedy decoding (`temperature=0.0`, `max_tokens=1024`), N=4 seeds per
cell. All scorers wired (standard + RAG-family).

- **H1 — Locals gain more from `kb_query` than cloud.** The local
  models benefit more than cloud models from having KB access. Formally:

  ```
  delta_local  = mean(end_state | with, local) − mean(end_state | without, local)
  delta_cloud  = mean(end_state | with, cloud) − mean(end_state | without, cloud)
  delta_local − delta_cloud ≥ 0.10
  ```

  with `local = {qwen3-14b-q4 (think:false), llama3.1-8b-q4}` and
  `cloud = {gpt-oss-20b-cloud, glm-5.1-cloud, gpt-oss-120b-cloud}`.

- **H2 — Models actually call `kb_query` when it's available.** When
  the tool is present, mean `tool_call_count` for `kb_query` per cell
  is **≥ 1.0** across *all* (model, task) cells in the `with`
  condition. (Operational sanity check — if a model never calls
  kb_query, the rest of H1/H3/H4 are moot for that model.)

- **H3 — Faithfulness improves with kb_query** on the one task in the
  v0.1 suite that opts into the faithfulness judge
  (`rag-bash-faithful-answer-shopt`, `include_faithfulness: true`,
  judge `gpt-oss-120b-cloud`). Formally:

  ```
  mean(faithfulness | with) − mean(faithfulness | without) ≥ 0.10
  ```

  computed over the 5 models × 4 seeds = 20 cells per condition on this
  task.

- **H4 — Without kb_query, hallucination is elevated on at least one
  retrieval task.** A task whose answer source is the KB (not
  pretraining) should see catastrophic failure when the model has to
  bluff. Operationalize as:

  ```
  ∃ (model, task) cell with task.success_predicate.type ∈ {
      retrieval_recall,
      workspace_file_contains
  }
  such that mean(end_state | without, model, task) ≤ 0.25.
  ```

  (i.e., on at least one cell in the `without` condition, the model
  fails on 3/4 seeds or worse.)

These four hypotheses are independent; each is judged on its own
evidence.

## Why this matters

1. **It's the lab's first measurement of RAG-conditioned agent
   behaviour.** EXP-002 told us locals can call tools accurately;
   EXP-003b tells us whether retrieval-tool access closes the
   tool-call-to-end-state gap that F-005 identified as the binding
   constraint on local models.
2. **It informs the local-first thesis at the RAG layer.** If H1
   confirms (locals gain more from KB access than cloud), the case for
   building local-first KB stacks gets stronger; if H1 is refuted, KB
   access is fungible across local/cloud and the bottleneck is
   elsewhere.
3. **It's the first end-to-end exercise of the full lab RAG stack.**
   The bash KB → `kb_query` MCP tool → Inspect Solver → RAG scorers →
   F-006 verdict pipeline. Anything that fails will fail loudly.

## Method

### Models (5 — same as EXP-002, same order)

| litellm_id | Backend | Notes |
|---|---|---|
| `qwen3-14b-q4` | local Ollama | reasoning **disabled** via `think: false`, per EXP-002 amendment |
| `llama3.1-8b-q4` | local Ollama | no reasoning mode |
| `gpt-oss-20b-cloud` | Ollama Cloud Pro | cheapest cloud reference |
| `glm-5.1-cloud` | Ollama Cloud Pro | top cloud tool-correctness in EXP-002 |
| `gpt-oss-120b-cloud` | Ollama Cloud Pro | upper bound for "spend more, do better" |

### Tasks (6 — `pbs-agent-rag-v0.1`)

The full PBS-Agent-RAG v0.1 suite (`tasks/pbs-agent-rag-v0.1/basics.yaml`):

| # | slug | predicate | judge? |
|---|---|---|---|
| 1 | rag-bash-redirection-operator | workspace_file_contains | no |
| 2 | rag-bash-param-expansion-forms | workspace_file_contains | yes (trajectory) |
| 3 | rag-bash-compare-test-bracket | workspace_file_contains | no |
| 4 | rag-bash-cite-section-for-arrays | workspace_file_exists | no (attribution) |
| 5 | rag-bash-for-loop-recall | retrieval_recall | no |
| 6 | rag-bash-faithful-answer-shopt | workspace_file_exists | yes (**faithfulness**) |

### Conditions (2)

- **with** — task's declared tool list, which includes `kb_query`
- **without** — task's tool list with `kb_query` removed (the rest —
  `fs_write` etc. — stays). Implemented as a sweep-level
  `tool_filter: exclude_kb_query` flag interpreted by the adapter.

### Seeds (4)

`[1, 2, 3, 4]`. EXP-002 used N=8; EXP-003b uses N=4 because (a) it's
exploratory and (b) cost (5 models × 6 tasks × 2 conditions × 8 = 480
already-known territory). N=4 still gives a usable pass⁴ on the
binary scorers; bootstrap CIs are reported.

### Total cells

`6 tasks × 5 models × 2 conditions × 4 seeds = 240 cells`.

### Config (1)

```yaml
name: greedy-1024
temperature: 0.0
top_p: 1.0
max_tokens: 1024
scaffold: single_turn   # runner dispatches to agent path on max_turns>1
```

### Pre-flight pilot (REQUIRED, runs BEFORE the full sweep)

`6 tasks × 5 models × 1 condition (with) × 1 seed = 30 cells`. Roughly
15-20 min wall. Sanity checks:

- All 30 cells finish `done`, not `error`
- Every cell with `kb_query` in tools actually has ≥ 1 `kb_query` call
  in the trajectory
- RAG scorers (`recall_at_k`, `mrr`, `ndcg`, `attribution`,
  `faithfulness`) populate the `agent_logs.turns->score_breakdown` for
  the appropriate tasks (and NOANSWER for the rest)
- The judge fires end-to-end at least once (task 6,
  `rag-bash-faithful-answer-shopt`)
- No harness regressions (sandbox boot, MCP stdio handshake, LiteLLM
  routing, KB mount) since EXP-002

Any task-design issues surface NOW (the F-005 lesson: 1 hour pilot
saves 4 hours of contaminated full-sweep data).

### Evaluators (pre-registered)

Deterministic, applied to every `done` run:

- `end_state` — task `success_predicate` (every task carries one)
- `tool_correctness` — `target_tool` from `task.rubric.tool_call`
- `budget_respected` — `actual_turns ≤ max_turns AND tool_call_count ≤ tool_budget`
- `recall_at_k` — for tasks with `retrieval_recall` predicate (task 5)
- `mrr` — same as above (task 5)
- `ndcg` — same as above (task 5)
- `attribution` — all RAG tasks (cheap, just regex over final message)

LLM-judge (selective):

- `trajectory_judge` — only when `include_judge: true` (task 2)
- `faithfulness` — only when `include_faithfulness: true` (task 6).
  Judge model: `gpt-oss-120b-cloud`. 20 judge calls per condition × 2
  conditions = **40 judge calls** for faithfulness, plus 20 + 20 = 40
  for trajectory_judge. **80 judge calls total.**

### Statistics

- `pass@1` and `pass^4` per (model, task, condition) on `end_state`.
- 95 % bootstrap CIs (n_resamples=2000) on cell pass rates and on
  faithfulness mean.
- Per-condition `delta_local` and `delta_cloud` as defined in H1.
- Per-(model, task) `kb_query` invocation count in the `with` condition
  for H2.

## Success / failure criteria

Each hypothesis is judged by the pre-registered rule, applied AFTER the
sweep + scoring complete. No peeking.

- **H1 confirmed** ⇔ `delta_local − delta_cloud ≥ 0.10`.
- **H2 confirmed** ⇔ for every (model, task) cell in the `with`
  condition, mean kb_query calls ≥ 1.0 (averaged over the 4 seeds).
- **H3 confirmed** ⇔ on task 6, `mean(faithfulness | with) −
  mean(faithfulness | without) ≥ 0.10`.
- **H4 confirmed** ⇔ at least one (model, task) cell in the `without`
  condition with the appropriate predicate type has
  `mean(end_state) ≤ 0.25`.

Any failure modes (error rate > 5 %, scorer NOANSWER > 20 % on cells
where the scorer should apply) are escalated in F-006 and reported as
UNDEFINED.

## Confounders to control

- **Same KB across all cells**: only the bash KB is exercised. RAG-on-X
  for X ≠ bash is out of scope.
- **One sandbox image**: `sandbox_image_hash` captured per run. Within-sweep
  drift guard (per 7b9aa46) fires immediately.
- **Same seed schedule** across all cells.
- **qwen3 runs with `think: false`** (EXP-002 amendment, holds the
  EXP-001b finding constant).
- **`kb_query`'s alpha and k are not varied** — defaults are
  `alpha=0.5`, `k=5`. EXP-003a is the place to vary those; here we
  control them.
- **Tool list is the only delta between `with` and `without`** — same
  `system`, same `input`, same `max_turns`, same `tool_budget`. The
  difference is purely "is kb_query in the tools list."

## Kill criteria

- **Cell error rate ≥ 5 %**: STOP, triage, do not continue silently.
- **Sandbox failure rate ≥ 10 %**: abort and escalate to gVisor.
- **kb_query NOANSWER on > 20 % of `with`-condition cells** where it
  should have applied: STOP — the KB mount is broken.
- **Judge unreachable**: continue (faithfulness will NOANSWER for
  affected cells, called out in F-006).

## Pre-mortem

It's 3 days from now and EXP-003b was a methodological failure. What
plausibly went wrong?

1. **The `without` condition is trivial — models bluff convincingly.**
   Mitigation: H4 captures this directly; bluff success on retrieval
   tasks means H1 is also refuted, both verdicts honest.
2. **kb_query mount broken on per-cell sandbox**. Mitigation: the
   pilot exercises this; if any pilot cell has 0 kb_query calls, we
   stop and diagnose before the full sweep.
3. **Task-design issues** (carryover from F-005). Mitigation: 30-cell
   pilot. F-005 showed each task-design bug was visible in the
   first 1-2 cells of that task; same pattern expected here.
4. **glm-5.1-cloud + kb_query passthrough subtly broken** —
   LiteLLM tool-call routing of `kb_query` (a fairly chatty tool with
   a complex JSON response) may dropout. Mitigation: tool_correctness
   scorer catches this; the verdict surface explicitly.

## Budget estimate

- 240 cells × ~4 turns × ~800 tok/turn ≈ 768K tokens
- Cloud cells: 5 × 6 × 4 × 3 = 360 cloud-cell-runs at ~3.2K tok each
  ≈ 1.1M cloud tokens
- 80 judge calls (cloud) at ~2K tok each = 160K extra
- Wall time estimate: ~45-90 min (similar gVisor overhead to EXP-002,
  smaller matrix)

## Outputs

- `analysis/EXP-003b/SUMMARY.md` — top-line verdicts + headline numbers
- `analysis/EXP-003b/verdicts.md` — per-hypothesis verdicts + tables
- `analysis/EXP-003b/per_model_condition.csv`
- `analysis/EXP-003b/per_cell_runs.csv`
- `analysis/EXP-003b/faithfulness_slice.csv`
- `analysis/EXP-003b/kb_query_invocations.csv`

Findings rolled into the combined F-006 (with EXP-003a).
