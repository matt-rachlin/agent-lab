---
doc_id: prompt-test-tool-use-system-v1
title: Test for tool_use_system_v1
zone: lab
kind: card
status: active
owner: m
created: 2026-05-27
last_updated: 2026-05-27
last_verified: 2026-05-27
tags: [lab, prompt, test]
---

# Test for tool_use_system_v1

```yaml
prompt_id: tool_use_system_v1
tests:
  - name: "http_fetch_on_url_get"
    input: "GET http://example.com/status.json and return the uptime field."
    expected_tool_calls: ["http_fetch"]
  - name: "fs_read_for_local_file"
    input: "Open /workspace/data.csv and show me the first line."
    expected_tool_calls: ["fs_read"]
  - name: "shell_exec_for_pipeline"
    input: "Use the shell to count lines in /workspace/log.txt."
    expected_tool_calls: ["shell_exec"]
```

Targets the http_fetch / fs_read / shell_exec discrimination — the
distinguishing trait of this prompt vs the plainer agent_system_v1.
