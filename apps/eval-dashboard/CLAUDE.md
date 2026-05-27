---
doc_id: dashboard-claude
title: 'apps/eval-dashboard: agent notes'
zone: lab
kind: claude
status: active
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
depends_on:
- kind: doc
  target: dashboard-readme
- kind: code
  target: lab:apps/eval-dashboard/lib/db.py
- kind: code
  target: lab:apps/eval-dashboard/lib/minio.py
- kind: code
  target: lab:apps/eval-dashboard/lib/services.py
- kind: code
  target: lab:apps/eval-dashboard/lib/docs.py
tags:
- streamlit
- dashboard
- phase-15.3
- decoupled
---

# apps/eval-dashboard - agent notes

## Hard rules

1. **Never import `lab.*` here.** The whole point of this app is to
   keep working when `src/lab/` or `packages/lab-*/` are being
   refactored. Use psycopg + boto3 + sqlite3 + redis directly.

2. **Render-on-failure.** Every helper that touches a network service
   must catch broadly and return an empty / sentinel value. The
   dashboard renders red dots, not stack traces.

3. **No writes.** Streamlit runs locally and the user is logged in as
   the DB owner. Any write tool would let the user accidentally
   mutate Postgres from a dashboard click. The catalog DB at
   `~/db/m/docs.db` is opened with `mode=ro`.

4. **Cache aggressively, expire fast.** `@st.cache_data(ttl=10..60)`
   on every Postgres helper. Sweep monitor uses ttl=10 plus
   `st.autorefresh` to feel live.

## Page layout convention

```python
# every page top:
import sys
from pathlib import Path
_APP_DIR = Path(__file__).resolve().parent.parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
from lib import db  # noqa: E402
```

This lets `streamlit run apps/eval-dashboard/Home.py` work from any
cwd because each page bootstraps its own import path.

## Adding a page

1. New file `pages/N_Name.py` (N controls sidebar order).
2. Use the bootstrap snippet above.
3. Add `@st.cache_data(ttl=...)` to every DB query.
4. Add a test in `tests/test_lib.py` for any new helper in `lib/`.
5. Update `README.md` directory tree.

## Common pitfalls

- Streamlit's `experimental_autorefresh` was renamed; we shim both
  names in `pages/4_Sweep_Monitor.py`.
- `streamlit-agraph` is a soft dependency: pages must fall back to a
  static list if the import fails (see `1_Findings.py`).
- DuckDB postgres_scanner is loaded lazily inside `lib/db.py`; never
  at module-import time (importing breaks `pytest` when the extension
  can't be downloaded offline).

## Phase 15.1 coordination

This app shares `pyproject.toml` and `justfile` with sibling 15.1. We
only ever append to the bottom of those files. Sibling 15.1 only
modifies the top sections (workspace declarations, db-migrate path).
No conflicts.
