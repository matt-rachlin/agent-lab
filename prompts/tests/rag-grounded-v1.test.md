---
doc_id: prompt-test-rag-grounded-v1
title: Test for rag_grounded_v1
zone: lab
kind: card
status: active
owner: m
created: 2026-05-27
last_updated: 2026-05-27
last_verified: 2026-05-27
tags: [lab, prompt, test]
---

# Test for rag_grounded_v1

```yaml
prompt_id: rag_grounded_v1
tests:
  - name: "kb_query_first_on_knowledge_question"
    input: "What is the bash redirection operator for stderr to stdout?"
    expected_tool_calls: ["kb_query"]
  - name: "no_paraphrase_url_on_citation_request"
    input: "Find a page about bash arrays and reply with the source_url verbatim."
    expected_tool_calls: ["kb_query"]
  - name: "admit_uncertainty_when_chunks_dont_cover"
    input: "Use kb_query for 'a totally fabricated bash feature' and answer."
    expected_tool_calls: ["kb_query"]
    expected_response_substring: "do not"
```

Verifies the kb_query-first reflex and the "admit uncertainty when
chunks don't cover" discipline. The third test uses
`expected_response_substring` (rather than just tool-call shape) to
catch a model that calls kb_query but then hallucinates anyway.
