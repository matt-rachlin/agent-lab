# lab-agent

Agent execution:
- `lab.agent.sandbox` — gVisor `runsc` wrapper, HF cache mount logic
- `lab.agent.tool_pool` — pooled MCP-style tool servers
- `lab.agent.tools.*` — built-in tools (kb_query depends on lab-rag)

## Gotchas
- `gvisor_available()` shells out to `runsc --version`; mock in tests.
- `kb_query` is the reason lab-agent depends on lab-rag.
- `tools/_common.py` defines the workspace path-escape guard used by all tools.
