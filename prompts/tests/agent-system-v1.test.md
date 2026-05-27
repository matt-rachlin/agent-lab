---
doc_id: prompt-test-agent-system-v1
title: Test for agent_system_v1
zone: lab
kind: card
status: active
owner: m
created: 2026-05-27
last_updated: 2026-05-27
last_verified: 2026-05-27
tags: [lab, prompt, test]
---

# Test for agent_system_v1

```yaml
prompt_id: agent_system_v1
tests:
  - name: "tool_use_intent_fs_read"
    input: "Read /workspace/note.txt and tell me what's in it."
    expected_tool_calls: ["fs_read"]
  - name: "tool_use_intent_python_eval"
    input: "Compute the sum of the first 100 positive integers using Python."
    expected_tool_calls: ["python_eval"]
  - name: "no_tool_when_unneeded"
    input: "Greet me in one short sentence."
    expected_tool_calls: []
```

Each test verifies that a model under this system prompt makes the
expected tool-call decisions on a few canonical inputs. Real-model runs
are manual; unit tests use a mock caller.
