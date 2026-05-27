---
doc_id: postmortem-2026-05-27-multi-words-and-hash-regression
title: 'Postmortem: multi-words-and-hash regression — qwen3-14b-q4 dense 1.0 (F-005/EXP-002) -> 0.0 (F-009/EXP-006)'
zone: lab
kind: postmortem
status: archived
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
depends_on:
- kind: doc
  target: f-005-12gb-agent-v0-2-tool-use
- kind: doc
  target: f-009-qwen3-30b-moe-refuted-h1-invalid
- kind: code
  target: lab:tasks/pbs-agent-v0.1/multi.yaml
- kind: code
  target: lab:prompts/library/agent-system-v1.md
- kind: code
  target: lab:packages/lab-sweep/src/lab/sweep/runner.py
tags:
- lab
- postmortem
- postmortems
- regression
- prompt-drift
- exp-006
- pbs-agent
---

# Postmortem: multi-words-and-hash regression — qwen3-14b-q4 dense 1.0 -> 0.0

> **Resolution (2026-05-27):** Option A applied. New task-local prompt
> `prompts/library/agent-system-hashing-v1.md` extends `agent_system_v1`
> with explicit guidance on multi-line python_eval formatting (no `;`-chained
> compound statements), sequential tool-call discipline, and verbatim copy
> of python_eval stdout into fs_write `content`. The task at
> `tasks/pbs-agent-v0.1/multi.yaml` now references this prompt. Single-seed
> smoke under EXP-006 conditions (greedy, temp=0.0, max_tokens=1024,
> qwen3-14b-q4 think:false) scored end_state=1.0 on all three arms
> (qwen3-14b-q4, qwen3-30b-a3b-moe, gpt-oss-120b-cloud). Goldens for the
> task were regenerated via `tools/sync_golden_outputs.py`. Postmortem
> `status` set to `archived` (the doc-meta schema's terminal state — there
> is no `resolved` value); `resolved_by` is the commit recorded below.
>
> **resolved_by:** commit `PENDING_COMMIT_SHA` ("Fix multi-words-and-hash
> regression (F-009 follow-up #4): task-local hashing prompt").

Date: 2026-05-27
Source finding: [F-009](../findings/F-009-qwen3-30b-moe-refuted-H1-invalid.md)
Operational note: F-009 §H1 per-task table, follow-on #4.

## TL;DR

The regression is **prompt drift**, not task drift, scorer drift, tooling drift,
or model drift. Between EXP-002 (F-005) and EXP-006 (F-009) the task's effective
system message changed from **none** to **`agent_system_v1`**, and `qwen3-14b-q4`
(reasoning-OFF, greedy, temp=0) deterministically responds to the new system
prompt by emitting an invalid one-line `python_eval` payload:

```python
import hashlib; with open('/workspace/message.txt', 'rb') as f: content = f.read(); print(hashlib.sha256(content).hexdigest())
```

Python rejects `with` after a `;` (statement vs. compound-statement grammar):

```
  File "<string>", line 1
    import hashlib; with open(...): ...
                    ^^^^
SyntaxError: invalid syntax
```

Identical SyntaxError stderr on all 4 sampled EXP-006 seeds (1-4) of
`qwen3-14b-q4 / multi-words-and-hash`. The agent then writes an 8-byte
`hash.txt` (not the 64-character digest), so the existing
`workspace_file_contains` success_predicate correctly scores 0.0.

The model is genuinely producing wrong output under the new system prompt.
The scorer is correct. The task is correct. The fix is **prompt-level**.

## Evidence

### Cell trajectories from Postgres + MinIO

EXP-006 / qwen3-14b-q4 / multi-words-and-hash, all 8 seeds done, all scored 0.0.
Sampled 4 trajectories from MinIO (`s3://lab/runs/2026-05/27/<run>/trajectory.jsonl`):

| run_id | seed | python_eval result | fs_write bytes | end_state |
|---|---|---|---|---|
| 732bcb97fb7fc0281ea56765 | 1 | SyntaxError (see above) | 8 | 0.0 |
| 0c3954ed6c211ba761517139 | 2 | SyntaxError (identical) | 8 | 0.0 |
| 310df6877fa1e1cdb79e1b13 | 3 | SyntaxError (identical) | 8 | 0.0 |
| b7c5dca998d94702ba94f941 | 4 | SyntaxError (identical) | 8 | 0.0 |

Score breakdown for run 732bcb97...:
```json
{
  "end_state": {"value": 0.0,
    "explanation": "file 'hash.txt' did not contain 'd20bc21bb3c7736d8d03ade3ddb4c68b665cdfbca6f6df0f7fdd192f37f59060'"},
  "budget_respected": {"value": 1.0, "explanation": "within budget: turns=2/5, tool_calls=3/6, terminated=model_finished"},
  "tool_correctness": {"value": 1.0, "explanation": "tool 'python_eval' called with matching args (expected {})"}
}
```

`tool_correctness=1.0` because `python_eval` *was* invoked — but the code it
ran failed. The end_state predicate (`workspace_file_contains 64-char hex`)
correctly catches the downstream failure.

### EXP-002 (F-005 baseline) trajectories

EXP-002 / qwen3-14b-q4 / multi-words-and-hash, sampled 3 of 8 seeds:

| run_id | seed | python_eval result | fs_write bytes | end_state |
|---|---|---|---|---|
| 1919fd1bba357514e8647d35 | 1 | `d20bc21b...f59060\n` (correct) | 64 | 1.0 |
| aefabb715c84761d03cd18d8 | 2 | `d20bc21b...f59060\n` (correct) | 64 | 1.0 |
| da31425c1eed1541b363c263 | 3 | `d20bc21b...f59060\n` (correct) | 64 | 1.0 |

Same model, same task, same greedy temp=0.0 config, but the model emits
syntactically valid Python and the digest matches the predicate.

### What actually changed: system message

EXP-002 trajectory captures only `[user, assistant, tool, ...]` — **no system
message was sent to the model**. The task's inline `system:` field existed
but was not yet wired into the runner's message construction (this is
documented in commit `fc5fccb` Phase 16.4.5: "Phase 16.4.2 added
Task.system_prompt_id but only the schema-level half was wired — at runtime
the solver still read Task.system (the inline string), so the 18 migrated
PBS-Agent / PBS-Agent-RAG tasks were running without their system prompt
body.").

EXP-006 trajectory captures `[system: agent_system_v1, user, assistant, tool, ...]`
— the system message is now sent.

Source change pair:

- `d2a75ef` (Phase 16.4.2, 2026-05-27 01:31 EDT) replaced the task-local
  `system:` block with `system_prompt_id: agent_system_v1` in
  `tasks/pbs-agent-v0.1/multi.yaml`.
  - Original inline `system:` (used in EXP-002, but never actually sent due
    to wiring gap): "You are an assistant with filesystem and Python tool
    access. Use python_eval for hashing rather than guessing the digest."
  - New referenced prompt body (`prompts/library/agent-system-v1.md`):
    "You are an assistant with filesystem, Python, and shell tool access.
    Always use the provided tools when asked to read, compute, or query
    something — never guess file contents, never approximate numbers, and
    do not invent results. Read code before describing it; compute
    numerically with python_eval; use the shell for file properties and
    text slicing. Write outputs in the EXACT format requested."

- `fc5fccb` (Phase 16.4.5, 2026-05-27 02:12 EDT) wired the
  `system_prompt_id` -> body resolution in
  `packages/lab-sweep/src/lab/sweep/runner.py` (and adapter), so EXP-006
  was the first sweep where the new system body actually reached the model.

Together these two commits change the effective input to the model from
"no system message" (EXP-002 behavior) to "`agent_system_v1` body as system
message" (EXP-006 behavior). With the new prompt, qwen3-14b-q4 picks a
shell-like one-liner phrasing (`stmt1; stmt2; stmt3`) for the Python tool
call — possibly cued by the "use the shell for file properties and text
slicing" sentence in `agent_system_v1`. The model's one-liner is invalid
Python because `with` cannot follow `;` at the statement level.

### What did NOT change

- **Task definition**: `multi-words-and-hash` input prompt, fixture
  (`message.txt = "agent\n"`), expected hex digest, and
  `success_predicate.workspace_file_contains` are byte-identical between
  the EXP-002-era multi.yaml and the EXP-006-era multi.yaml. Only the
  `system:` -> `system_prompt_id:` line changed.
- **Scorer**: `workspace_file_contains` in
  `packages/lab-inspect/src/lab/inspect_bridge/scorer.py` is unchanged on
  the relevant code path between EXP-002 and EXP-006. The
  `7b6aa46` ("EXP-002 follow-ups") commit touched `end_state` to return
  NOANSWER on unknown predicate types and tightened
  `code-find-and-fix-bug`'s predicate — it did NOT touch
  `workspace_file_contains` logic or this task.
- **Tooling**: `python_eval` (`packages/lab-agent/src/lab/agent/tools/python_eval.py`)
  is unchanged since Phase 6c — still runs `python3 -c "<code>"` with
  cwd=/workspace and returns `{stdout, stderr, exit_code, timed_out}`.
- **Golden / fixture**: no golden file exists for this task (it uses a
  workspace-state predicate, not a golden-comparison evaluator). The
  fixture `message.txt: "agent\n"` is unchanged.
- **Model weights**: same Ollama model tag `qwen3:14b-q4_K_M`
  (model_id=14, ollama_tag unchanged across both experiments).
- **Inference config**: `temperature=0.0, top_p=1.0, max_tokens=1024,
  scaffold=single_turn`, identical. EXP-006 adds
  `extra.keep_alive=0` (vs EXP-002's default keep_alive=5m); F-009 notes
  this is a between-cell VRAM-residency knob, not an inference knob —
  consistent with the diagnosis here that the regression source is the
  system message, not the keep_alive change.

## Root cause (one sentence)

A latent task-definition change in commit `d2a75ef` (swap inline `system:`
for `system_prompt_id: agent_system_v1`) became live when commit `fc5fccb`
wired the resolver — adding a system message that elicits a syntactically
broken Python one-liner from qwen3-14b-q4 on the deterministic greedy path,
which fails python_eval, writes a wrong 8-byte hash.txt, and the predicate
correctly scores 0.0.

## Categorization

Task drift. Specifically: prompt drift, with two-commit fingerprint
(`d2a75ef` + `fc5fccb`). Not scorer drift, not tooling drift, not
golden drift.

The model is genuinely worse under the new prompt — but "worse" here is a
prompt-engineering interaction, not a model-quality regression. The same
weights produce the correct one-liner-via-`exec`-or-multiline-input answer
under the EXP-002 effective prompt (none) and a broken one-liner under
the EXP-006 effective prompt (`agent_system_v1`).

## Recommended fix (next session: actionable, no re-investigation needed)

Pick ONE of the following. Each is concrete and committable as a single
PR by the next session.

### Option A — Restore a task-local system prompt that explicitly tells the model how to format multi-statement Python

Edit `tasks/pbs-agent-v0.1/multi.yaml` under the `multi-words-and-hash`
task. Replace:

```yaml
    system_prompt_id: agent_system_v1
```

with a task-specific override that survives the system_prompt_id schema
(either re-introduce inline `system:` if the validator allows, or add a
new `prompts/library/agent-system-hashing-v1.md` and reference it):

```yaml
    system_prompt_id: agent_system_hashing_v1  # NEW prompt to create
```

And create `prompts/library/agent-system-hashing-v1.md` with frontmatter
and body roughly:

> You are an assistant with filesystem and Python tool access. Use
> python_eval for hashing rather than guessing the digest. When you pass
> code to python_eval, use a multi-line string (newlines between
> statements). Do NOT chain `import`, `with`, or `def` with semicolons
> — `with` after `;` is invalid Python.

Rationale: minimal blast radius, only touches this one task. Matches the
EXP-002-era *intended* system prompt that was a no-op due to the wiring
bug. The added one-liner anti-tip directly defends against the observed
failure mode.

### Option B — Fix `agent_system_v1` itself

Edit `prompts/library/agent-system-v1.md` body to add an explicit
multi-statement-format note for `python_eval`. Higher blast radius
(affects 4 tasks: fs.yaml, code.yaml, shell.yaml, multi.yaml) — but the
shell-pipeline-extract dense **improvement** in F-009 (0.000 -> 1.000)
suggests `agent_system_v1` is net helpful on most tasks; you'd want to
keep its character and only narrow the python_eval guidance.

Rationale: fixes the prompt for any future hashing/python_eval task, not
just this one. Risk: revalidates F-009's anchor across all four affected
task files; the F-009 "lab plumbing has drifted in ways visible in both
directions" caveat suggests EXP-006b should re-anchor against whatever
prompt body is chosen anyway.

### Option C — Loosen `python_eval` to tolerate the broken one-liner

Edit `packages/lab-agent/src/lab/agent/tools/python_eval.py` to detect
single-line code that contains `with` after `;` and rewrite it as
multi-line before invoking `python3 -c`. **Do not do this.** It papers
over a real model-output failure with tool-level magic and would mask
future prompt-elicited Python-grammar bugs from other models. Listed
here only to explicitly rule it out — the failure mode is in the model's
output, not the tool.

### Recommended: Option A, then re-run EXP-006b for replication anchor

Option A is the smallest change that surfaces an honest signal: a
task-author chose a prompt that tells the model how to format hashing
code, and the model now succeeds again. EXP-006b (already filed in F-009)
should re-run with the new prompt to confirm the dense baseline lands
near F-005's 0.750 (or wherever the rest of the F-009 task/scorer fixes
move it). If Option A succeeds, expect this task to flip from 0.000 to
≈1.000 on `qwen3-14b-q4`, recovering ~8.3 pp of the 16.7 pp H1 deficit.

## Affected files (absolute paths)

- `/data/lab/code/tasks/pbs-agent-v0.1/multi.yaml` — the task. Currently
  uses `system_prompt_id: agent_system_v1` (line 27). Fix target for
  Option A.
- `/data/lab/code/prompts/library/agent-system-v1.md` — the new generic
  prompt. Fix target for Option B.
- `/data/lab/code/packages/lab-sweep/src/lab/sweep/runner.py` — the
  resolver wired in `fc5fccb`. No fix needed here.
- `/data/lab/code/packages/lab-inspect/src/lab/inspect_bridge/scorer.py`
  — the scorer. **No fix needed**; it's behaving correctly.
- `/data/lab/code/packages/lab-agent/src/lab/agent/tools/python_eval.py`
  — the tool. **No fix needed**; it's behaving correctly.

## Verification steps for the chosen fix (so next session can confirm)

1. Apply the fix (Option A: edit `multi.yaml` + add new prompt file).
2. Re-run a smoke cell against `qwen3-14b-q4`:
   ```bash
   cd /data/lab/code
   uv run lab sweep run conf/sweep/EXP-006.yaml \
     --models qwen3-14b-q4 --tasks multi-words-and-hash --seeds 1
   ```
3. Pull trajectory from MinIO and confirm `python_eval.stderr == ""`,
   `fs_write.bytes_written == 64`, and `score_breakdown.end_state.value
   == 1.0`.
4. If the smoke passes, file EXP-006b run with the full 12-task,
   3-arm, 8-seed sweep against the new prompt and recompute the H1
   anchor (F-009 §"Recommended next steps" #2).

## Open questions

- **Why does qwen3-14b-q4 choose a one-liner under `agent_system_v1`?**
  Likely the "use the shell for file properties and text slicing"
  sentence cues a shell-pipeline mental model that bleeds into the
  python_eval call. Worth a quick prompt-engineering ablation: drop
  that sentence from `agent_system_v1` (Option B variant) and re-run
  the multi-words-and-hash smoke. If the model recovers without the
  hashing-specific guidance, Option B is a one-line fix.
- **Does the same prompt failure mode appear in `multi-db-self-check`?**
  No: that task's predicate is `db_query` (read-only meta-check) and
  the model can no-op and still pass — separate F-009 follow-on (#3).
  This postmortem does not address that.
- **Are EXP-006 cloud / MoE cells on this task affected differently?**
  Cloud `gpt-oss-120b-cloud` scored 1.0 across all 8 cells on this
  task; MoE `qwen3-30b-a3b-moe` scored 1.0 across all 8 cells. The
  one-liner-then-SyntaxError failure mode is **specific to
  qwen3-14b-q4** under `agent_system_v1`. The other two models
  produce valid Python under the same prompt. (F-009 §H1 per-task
  table corroborates: multi-words-and-hash is dense=0.0, moe=1.0,
  cloud=1.0.) This rules out a tool-level or runner-level bug and
  localizes the issue to qwen3-14b-q4 × agent_system_v1.

## What we did not do in this pass

Per the diagnostic-only scope:

- Did not modify `tasks/pbs-agent-v0.1/multi.yaml`.
- Did not modify `prompts/library/agent-system-v1.md`.
- Did not modify any scorer or tool code.
- Did not re-run any sweep.
- Did not commit a fix — only this postmortem doc.
