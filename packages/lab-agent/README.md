# lab-agent

Agent runtime: gVisor-isolated `Sandbox`, `ToolPool` for concurrent tool servers,
and the in-tree tool servers (`fs_read`, `fs_write`, `fs_grep`, `http_fetch`,
`kb_query`, `python_eval`, `shell_exec`).

Namespace: `lab.agent.*`. Depends on lab-core, lab-rag.
