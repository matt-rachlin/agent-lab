---
doc_id: lab-session-start
title: Runbook — lab SessionStart hook
zone: lab
kind: runbook
status: active
owner: m
created: '2026-05-26'
last_updated: '2026-05-26'
last_verified: '2026-05-26'
tags:
- lab
- runbook
- runbooks
---
# Runbook — lab SessionStart hook

This runbook documents the lab-specific Claude Code SessionStart hook that
prints a one-screen status summary when a session opens under `~/lab/` or
`/data/lab/`.

## What it does

Six lines, prefixed `[lab]`:

1. Active experiment — most recent `experiments` row with `status='running'`
2. Last 3 findings (by `finding_id` desc until an `importance` column exists)
3. Running sweep PIDs from `/data/lab/services/sweep-pids/*.pid`
4. GPU lease state (Valkey key `lab:gpu:lease:0` + TTL)
5. Rerank service status (`curl /healthz` on the configured URL)
6. doc-graph placeholder — `Phase 14 pending`

The hook is silent (exit 0, no output) for any cwd outside `~/lab/` or
`/data/lab/`, so it doesn't pollute non-lab sessions.

## Where it lives

Script: `/data/lab/code/scripts/lab-session-start.sh` (in this repo).

Install location (NOT in this repo): the Claude Code user settings file
at `~/.claude/settings.json`, under `hooks.SessionStart`, as a new
entry alongside the existing color/rename hooks. The entry looks like:

```json
{
  "hooks": [
    {
      "type": "command",
      "command": "/data/lab/code/scripts/lab-session-start.sh 2>&1 | awk 'NF{print}' | head -10",
      "statusMessage": "lab session start"
    }
  ]
}
```

## Install (one-time)

The hook was installed on 2026-05-26 by Phase 13.7. To re-install on a
fresh box, append the above JSON object to the
`hooks.SessionStart` array in `~/.claude/settings.json`. Example
idempotent installer:

```bash
python3 - << 'PY'
import json
p = '/home/m/.claude/settings.json'
with open(p) as f: d = json.load(f)
ss = d.setdefault('hooks', {}).setdefault('SessionStart', [])
already = any(
    'lab-session-start.sh' in h.get('command', '')
    for entry in ss for h in entry.get('hooks', [])
)
if not already:
    ss.append({"hooks": [{
        "type": "command",
        "command": "/data/lab/code/scripts/lab-session-start.sh 2>&1 | awk 'NF{print}' | head -10",
        "statusMessage": "lab session start"
    }]})
    with open(p, 'w') as f: json.dump(d, f, indent=2)
    print("installed")
else:
    print("already installed")
PY
```

## Environment variables

The script honours these overrides (default in parentheses):

- `LAB_PG_DSN` (`postgresql://m@/lab`)
- `LAB_RERANK_URL` (`http://127.0.0.1:8401`)
- `LAB_SWEEP_PIDS_DIR` (`/data/lab/services/sweep-pids`)
- `LAB_REDIS_URL` (`redis://localhost:6379/0`)
- `LAB_HOOK_TIMEOUT` (`1` — per-subcommand timeout in seconds)

## Verifying

```bash
cd /data/lab/code
./scripts/lab-session-start.sh
```

Expected output (5-line sample, from 2026-05-26):

```
[lab] active experiment: EXP-003b
[lab] last 3 findings: F-007|high F-006|high F-005|medium
[lab] sweeps running: (none)
[lab] gpu lease: (free)
[lab] rerank service: up (http://127.0.0.1:8401)
[lab] doc-graph: Phase 14 pending
```

From a non-lab cwd:

```bash
cd ~/workspace && /data/lab/code/scripts/lab-session-start.sh
# (no output, exit 0)
```

## Future migration

Phase 14 lands a `~/system/configs/claude/workspaces.json` manifest that
the workspace-level SessionStart hook walks. When that lands, this
hook's install location moves into the manifest and the entry in
`~/.claude/settings.json` is removed.
