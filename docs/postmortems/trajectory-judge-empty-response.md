---
doc_id: postmortem-trajectory-judge-empty-response
title: 'Postmortem: trajectory judge empty-response on EXP-003b gpt-oss-20b cells'
zone: lab
kind: postmortem
status: active
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
tags:
- lab
- postmortem
- postmortems
- judge
- rag
- exp-003b
---

# Postmortem: trajectory judge empty-response on EXP-003b gpt-oss-20b cells

Date: 2026-05-27
Source experiment: [EXP-003b](../exp/EXP-003b.md)
Source finding: [F-006](../findings/F-006-rag-hybrid-wins-locals-need-kb.md)
Tracker entry: lab master roadmap §17.6 (task #32).
Reproduction artifacts: `/tmp/lab-17.6/` (transient).

## What happened

EXP-003b's `faithfulness` scorer (LLM-as-judge over the agent's final
response and retrieved chunks, judge model = `gpt-oss-120b-cloud`)
returned 5/20 NOANSWERs across the with-kb cells. F-006 had attributed
4/5 to structural `kb_query`-never-called cases (llama3.1) — those are
real NOANSWERs from `_kb_query_calls()` returning empty. The
fifth was reported as "1/4 gpt-oss-20b-cloud (kb_query called but
judge returned empty)" and was the only **unexplained** case.

Re-examining the data turned up not 1 but **3 gpt-oss-20b cells** with
`{"value": 0.0, "explanation": "empty judge response"}` (`value=0.0`
rather than `null`, so they show up as zeros in the mean and weren't
counted as NOANSWERs in F-006's bucket). All three cells were on the
same task — `rag-bash-faithful-answer-shopt`. A fourth cell on the
same task NOANSWERed for a different reason ("kb_query calls returned
no chunk text").

| run_id              | task                             | seed | faithfulness                                          |
| ------------------- | -------------------------------- | ---- | ----------------------------------------------------- |
| `7301501c…f239d`    | rag-bash-faithful-answer-shopt   | 1    | 0.0, "empty judge response"                           |
| `507c5e5a…295f`    | rag-bash-faithful-answer-shopt   | 2    | null, "kb_query calls returned no chunk text"         |
| `9e72124b…f8c4`    | rag-bash-faithful-answer-shopt   | 3    | 0.0, "empty judge response"                           |
| `aca86fd2…7eed`    | rag-bash-faithful-answer-shopt   | 4    | 0.0, "empty judge response"                           |

## Root cause

**Classification: (b) prompt/config issue (`max_tokens` too low for a
reasoning judge model).**

`lab.eval.judge._call_litellm` hard-codes `max_tokens=256` on every
judge request:

```python
# packages/lab-eval/src/lab/eval/judge.py:90
"max_tokens": 256,
```

`gpt-oss-120b-cloud` is a reasoning model. Its API response splits
output between an internal `reasoning_content` field (chain-of-thought,
unreturned in `message.content`) and the visible `content`. The judge
client reads only `content`. When `max_tokens=256` the model spends
all 256 completion tokens on `reasoning_content`, hits `finish_reason:
length`, and emits the response with `content=""`. `parse_judge_response`
then maps empty text to `(0.0, "empty judge response")` and
`_normalise_1_to_5(0.0, "empty…")` clamps to `0.0`.

Reproduction (manual `curl` to LiteLLM with the exact rebuilt prompt
from cell `7301501c…f239d`):

```
max_tokens=256 :  finish_reason=length, content_len=0,
                  reasoning_len=1066, completion_tokens=256
max_tokens=2048:  finish_reason=stop,   content="{\"score\": 5, …}",
                  reasoning_len=1240,   completion_tokens=290
```

All three empty-response cells reproduce identically — at
`max_tokens=2048` they return well-formed JSON scores of 1, 4, and 5
respectively. The reasoning trace is ~250–470 completion tokens; the
visible JSON is ~30–40 tokens. A `max_tokens` budget below ~300 is a
guaranteed empty-content trap for this judge model.

### Why classifications (a) and (c) are ruled out

- (a) Judge model degenerate: rejected — the model produces correct
  JSON on retry with a higher token budget, identical prompt, same
  temperature.
- (c) Parser bug: rejected — `parse_judge_response` handles the empty
  string correctly (returns `(0.0, "empty judge response")`). The
  issue is upstream — there is no visible content to parse.

### Why F-006 caught only one cell

F-006's tally was "5/20 NOANSWER" using the breakdown's `value: null`
sentinel. The empty-response failure mode writes `value: 0.0` (a real
score), not `null`, so the three empty-judge cells were counted as
"unfaithful score 0.0" in the with-kb arm rather than as NOANSWERs.
The author's "1/4 … judge returned empty" appears to have come from
manual spot-checking, not a systematic query. Net effect on F-006's
verdict: the with-kb faithfulness mean is depressed by three spurious
0.0s out of 20, and the NOANSWER bucket was undercounted by 3. F-006's
qualitative verdict ("UNDEFINED — data, not failure") still stands —
the experiment was already pre-registered to treat the 5 with-kb
NOANSWERs as undefined.

## Fix

Two changes in `packages/lab-eval/src/lab/eval/judge.py`:

1. Raise the default `max_tokens` from 256 → 1024 to leave room for
   reasoning models' CoT plus the JSON tail.
2. When `message.content` is empty but the response has a non-empty
   `reasoning_content`, attempt to extract a JSON object from the
   reasoning before falling back to the empty-response sentinel — this
   recovers cleanly for any future judge model with the same shape
   even at low `max_tokens`.

The fix is small enough to ship with this postmortem.

## Lessons

- **Reasoning models eat the token budget invisibly.** Any LLM-as-judge
  built before reasoning models existed needs its `max_tokens` audit
  re-done. The other judge call site (`lab.inspect_bridge.scorer`)
  routes through this same function, so the fix covers both
  faithfulness and trajectory_judge.
- **`value: 0.0` vs `value: null` is a meaningful distinction.** F-006's
  analysis should treat `score=0.0` with explanation containing
  `"empty judge response"` as NOANSWER, not 0.0. The analyzer doesn't
  currently do that. Filed as a small follow-up: extend
  `scripts/analyze_exp003b.py` (or its successor in `lab.analyze.*`)
  to recognise the empty-judge sentinel and bucket those cells as
  NOANSWER. Not critical — F-006's qualitative verdict was already
  conservative.

## Tracker resolution

Master roadmap §17.6 (task #32 — "Trajectory judge bug investigation"):
**resolved** as of this commit. Root cause was a token-budget config
mismatch with reasoning judge models, not a judge degeneracy and not
a parser bug. The fix is shipped in the same commit as this
postmortem.

## Verified

- 2026-05-27: 3/3 affected cells re-judged manually with
  `max_tokens=2048`, all returned well-formed `{"score": …,
  "reasoning": …}` JSON. Repro recipe is documented above; the
  artifacts (`req.json`, `resp.json`, `judge-prompt.txt`) were under
  `/tmp/lab-17.6/` during the investigation and intentionally not
  committed.
