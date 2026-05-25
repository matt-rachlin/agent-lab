# EXP-NNN: <one-line title>

Date created: YYYY-MM-DD
Status: planned
Pre-registered: <fill in git commit sha when this file is committed>

## Hypothesis
<falsifiable prediction, e.g. "Qwen3-14B-Q4 at temp=0 will score within 5pp of Qwen3-14B-Q5 on PBS-coding tasks">

## Why this matters
<the decision this experiment informs>

## Method
- Model(s):
- Quantization(s):
- Backend (Ollama / vLLM / llama.cpp):
- Tasks (which PBS subset, or external benchmark):
- Configs (temperature, top_p, scaffold variant, retry policy):
- Eval metrics (pre-registered):
- Seeds: <list, ≥ 8>
- Judges (if any) + oracle slice plan:

## Success / failure criteria
<defined BEFORE running, with the exact threshold + test>

## Confounders to control
<which axes are held fixed, which are varied, which are noisy>

## Kill criteria
<conditions under which we abort early>

## Pre-mortem
Imagine it's <end date> and this experiment failed badly. List plausible reasons + cheap mitigations now.

## Estimated cost
GPU-hours: ... | Cloud calls: ... | Wall time: ...
