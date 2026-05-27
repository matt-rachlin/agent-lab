---
doc_id: evals-golden-readme
title: Frozen golden outputs — regression captures
zone: lab
kind: readme
status: active
owner: m
created: 2026-05-27
last_updated: 2026-05-27
last_verified: 2026-05-27
tags: [lab, evals, golden, regression]
---

# evals/golden/ — frozen golden outputs

Captured (suite, task, model) trajectories used as regression anchors. The
schema and comparator live in `lab.eval.golden`; the (re-)generation tool
is `tools/sync_golden_outputs.py`.

## Layout

```
evals/golden/
├── README.md                    (this file)
├── pbs-v0.1/
│   ├── math-001/
│   │   ├── qwen3-14b-q4.json
│   │   └── gpt-oss-120b-cloud.json
│   └── ...
├── pbs-agent-v0.1/
│   └── ...
└── pbs-agent-rag-v0.1/
    └── ...
```

One JSON file per `(suite, task_slug, model)` triple. Filenames are
`<model>.json` and live under `<suite>/<task_slug>/`.

## File schema

```json
{
  "task_slug": "math-001",
  "model": "qwen3-14b-q4",
  "suite": "pbs-v0.1",
  "config_hash": "521306be5bf94943",
  "captured_at": "2026-05-27T10:30:00Z",
  "response_text": "148",
  "tool_calls": [],
  "scorer_outcomes": {
    "exact_match": 1.0
  }
}
```

* `config_hash` — hash of the model + sweep config used to capture. If a
  later replay yields a different hash, the comparator still works but
  callers should treat it as a different environment.
* `captured_at` — ISO 8601 UTC timestamp.
* `response_text` — the final assistant turn (verbatim).
* `tool_calls` — flattened `[{tool, args}]`, one entry per call across all
  turns, in chronological order.
* `scorer_outcomes` — `{scorer_name: float}` for every scorer that ran.

## Regenerating

```bash
# Dry-run: list what would be generated, don't run any model
uv run python tools/sync_golden_outputs.py --dry-run

# Capture goldens for one suite + model (real model run)
uv run python tools/sync_golden_outputs.py \
    --suite pbs-v0.1 --model qwen3-14b-q4

# Force overwrite existing
uv run python tools/sync_golden_outputs.py \
    --suite pbs-v0.1 --model qwen3-14b-q4 --force
```

By default the tool **skips** existing files; use `--force` to overwrite.
It also respects `lab:gpu:lease:0` — if the lease is held by another
process the script refuses to launch real model runs and exits with a
hint. Use `--dry-run` to bypass the lease check.

## Coverage targets

The initial population (planned, not yet captured at time of writing):

| Suite              | Models                                  | Tasks | Files |
| ------------------ | --------------------------------------- | ----- | ----- |
| pbs-v0.1           | qwen3-14b-q4, gpt-oss-120b-cloud        | 24    | ~48   |
| pbs-agent-v0.1     | llama3.1-8b-q4, gpt-oss-120b-cloud      | 12    | ~24   |
| pbs-agent-rag-v0.1 | glm-5.1-cloud                           | 6     | ~6    |

Total: roughly 78 goldens. Wall: 30-60 min on free GPU.

## Comparator

`lab.eval.golden.compare_to_golden(...)` reads the file, compares against
an actual trajectory dict, returns a `GoldenComparison` indicating

* `same_response` (str equality on `response_text`)
* `same_tool_calls` (list-of-`{tool, args}` equality)
* `scorer_drift` (dict of `{scorer_name: |delta|}` exceeding tolerance)

If the golden file is missing, `found=False` and callers decide whether to
fail or warn.
