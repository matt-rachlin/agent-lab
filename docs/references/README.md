# References

Library of papers and posts that inform the lab's methodology. Two parts:

- `references.bib` — BibTeX for tools that want it
- Per-paper notes — `<author><year><shorttitle>.md`, parsed by `lab.references.sync` (planned, Phase 6+)

## Adding a paper

1. Run `lab refs new <key>` (planned) OR copy `_template.md` to `docs/references/<your-key>.md`
2. Fill in the front matter (cite key, title, authors, year, venue, url)
3. Add a `## Why I read this` (one paragraph: what question of mine it answered)
4. Add a `## Key claims` (the 3–5 things from the paper you actually believe)
5. Add a `## Reservations` (where you'd push back, or where the methodology is shaky)
6. Add the BibTeX entry to `references.bib`

## Per-paper note template

[`_template.md`](./_template.md) — start every new paper note from this.

## Existing entries (seeded 2026-05-25)

- [sierra2024-tau2-bench](./sierra2024-tau2-bench.md) — Sierra `τ²-bench` & the pass^k metric
- [tan2024-judgebench](./tan2024-judgebench.md) — LLM judges as evaluators (limits)
- [mehta2025-clear](./mehta2025-clear.md) — CLEAR multi-dimensional evaluation framework
- [cje2025-causal-judge](./cje2025-causal-judge.md) — calibrating cheap judges with an oracle slice
- [bjarnason2026-randomness](./bjarnason2026-randomness.md) — variance in agentic evals at temp=0
- [pineau2020-reproducibility](./pineau2020-reproducibility.md) — NeurIPS reproducibility checklist
- [breck2017-ml-test-score](./breck2017-ml-test-score.md) — 28-item production-readiness rubric
- [karpathy2019-recipe](./karpathy2019-recipe.md) — A recipe for training neural networks
