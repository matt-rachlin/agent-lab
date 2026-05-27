---
doc_id: contamination-check
title: 'Protocol: Benchmark contamination check'
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
- protocols
---
# Protocol: Benchmark contamination check

**Before any task is added to PBS (or any external benchmark is used in the lab), do a contamination check and record the result.**

Contamination = the model has seen the task (or close paraphrases) during training. Famous example: SWE-bench Verified shows 81% on contaminated models that drop to 46% on SWE-bench Pro for the same model ([Morph LLM analysis](https://www.morphllm.com/swe-bench-pro)).

For a personal benchmark built in 2026, contamination affects everything from 2024-pretrained models forward. We don't get to ignore this.

---

## 1. Define the task uniquely

Before testing, the task should be:

- **Novel phrasing**, not a copy of a well-known prompt
- **Specific answer**, not a guessable common-knowledge fact
- **Dated after the model's likely training cutoff** if it depends on facts

For math tasks: change the numbers (e.g. `(47 * 8) - (12 * 19)` instead of textbook values like `(2 * 3) + (4 * 5)`).

For knowledge tasks: pick obscure-but-verifiable facts that are unlikely to appear verbatim in pretraining corpora.

## 2. The "verbatim" check

For each candidate task, search the open web for the exact prompt:

```bash
# Quick check via DuckDuckGo HTML — replace with your favorite engine
curl -s 'https://duckduckgo.com/html/?q=%22Compute+%2847+%2A+8%29+-+%2812+%2A+19%29%22' \
    | grep -c 'result__'   # 0 hits = good, >0 = look at them
```

If the exact prompt appears in a public corpus (Stack Overflow, textbooks, Common Crawl-indexed), assume it's contaminated.

## 3. The "Oren shuffle" check ([Oren et al. 2023](https://hf.co/papers/2310.17623))

For a model to "know" a task during pretraining, the order of options/choices in the original document is memorised. If you shuffle the options and the model's accuracy drops significantly, the original was likely in training data.

Applies to multiple-choice tasks; the lab's PBS v0.1 is mostly free-form so this is N/A here, but it's the right check for adding any external benchmark (BFCL, MMLU subsets, etc.).

## 4. The "dates after cutoff" technique

Tasks whose answers depend on events after a model's training cutoff can't be contaminated.

For the lab: ensure at least 5 PBS tasks reference 2024+ facts to detect future contamination drift. Mark them in the task `payload.metadata.contamination_anchor: true`.

## 5. The MMLU-CF-style perturbation check ([Xu et al. 2024](https://hf.co/papers/2412.15194))

Generate a "perturbed" twin of the task — same difficulty, different surface form, same answer. If the model's accuracy is significantly higher on the original than the twin, contamination is likely.

For PBS, add perturbed twins to tasks where contamination is suspected. The lab's task schema supports this: same `slug` with a `_twin` suffix, `category` set to indicate it.

## 6. Record the result

Every PBS task that has passed a contamination check should have, in its YAML:

```yaml
metadata:
  contamination_check:
    method: verbatim_web_search | oren_shuffle | dates_after_cutoff | perturbed_twin
    checked_at: 2026-05-25
    result: clean | likely_contaminated | unknown
    notes: "0 web hits as of 2026-05-25 for the exact prompt"
```

`likely_contaminated` doesn't mean drop the task — it means the lab marks any claim resting on that task with the contamination caveat.

## 7. When to re-check

- When a new model line is released (its training cutoff may be later than your last check)
- Quarterly, for any task that drives published claims
- Whenever a model unexpectedly scores 100% on a task we thought was hard

## 8. PBS v0.1 status (as of 2026-05-25)

PBS v0.1 was built without formal contamination checks. **All claims derived from PBS v0.1 carry the caveat: "contamination not formally checked at v0.1; partial recheck planned before Phase 5 EXP-001 execution."**

When PBS gets refreshed for EXP-001 (Phase 5), every task gets a contamination check per this protocol before going into the v0.2 release.
