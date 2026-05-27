---
doc_id: prompts-readme
title: Prompts library — canonical prompts referenced by tasks
zone: lab
kind: readme
status: active
owner: m
created: 2026-05-27
last_updated: 2026-05-27
last_verified: 2026-05-27
tags: [lab, prompts]
---

# prompts/ — canonical prompts library

Task YAMLs reference these prompts by ID (`system_prompt_id: agent_system_v1`)
rather than inlining them. The registry lives in `lab.eval.prompts`; golden
tests per prompt live in `prompts/tests/`.

## Convention

Each prompt file:

* lives at `prompts/library/<prompt_id>.md`
* has doc-meta frontmatter (`kind: prompt`)
* declares `prompt_id` and `version` in its frontmatter
* contains the prompt body after the frontmatter close

A revision is either a `version: N` bump (overwrites the previous body) or
a renamed file `<prompt_id>_v2.md` (keeps both). Use the rename path when
old runs in the DB should still resolve their original prompt.

## Frontmatter

```yaml
---
doc_id: prompt-agent-system-v1
title: Agent system prompt v1
zone: lab
kind: prompt
status: active
owner: m
created: 2026-05-27
last_updated: 2026-05-27
last_verified: 2026-05-27
tags: [lab, prompt, agent, system]
---
```

The body that follows is the prompt text. No further parsing — what you
write is what the registry returns.

A `prompt_id` line is parsed out of the YAML even though it isn't in the
strict doc-meta schema. To stay schema-compliant we use the doc_id as the
`prompt_id` source-of-truth: strip the leading `prompt-` and the trailing
`-vN` for the lookup id, then carry the version separately. So:

```
doc_id: prompt-agent-system-v1
   ↓
prompt_id: agent_system_v1
version:   1
```

## Versioning

The default lookup `registry.get("agent_system_v1")` returns the highest
known version. To pin: `registry.get("agent_system_v1", version=1)`.

## Tests

Each prompt has a corresponding `prompts/tests/<prompt_id>.test.md` golden
test file. See `lab.eval.prompts.test_runner` for the test format. Tests
run via:

```bash
uv run python -m lab.eval.prompts test [--prompt-id ...]
```

In CI the tests run with a mocked model; real-model runs are manual.
