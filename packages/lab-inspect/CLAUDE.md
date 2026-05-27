# lab-inspect

Inspect AI bridge:
- `lab.inspect_bridge.adapter` — convert lab `TaskRow` → Inspect `Task`
- `lab.inspect_bridge.solver` — model-with-tools loop
- `lab.inspect_bridge.scorer` — built-in lab scorers as Inspect scorers
- `lab.inspect_bridge.scorers.rag` — RAG-specific scorers (MRR, NDCG, recall@k, attribution, faithfulness)
- `lab.inspect_bridge.tools` — sandbox-backed tool registration for Inspect
- `lab.inspect_bridge.logwriter` — write Inspect logs back to lab manifests

## Gotchas
- `tools.py` invokes the sandbox via `_invoke_tool_via_sandbox_sync`; mocks are heavy.
- `solver.py:_truncate` is private — used by tests.
