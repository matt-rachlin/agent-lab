# lab-core

Foundation package. Holds shared primitives every other lab-* package depends on:

- `lab.core.settings` — env-driven config (pydantic-settings)
- `lab.core.manifest` — capture run manifests (HW, deps, git)
- `lab.core.notify` — ntfy helpers
- `lab.core.gpu_lease` — Valkey-backed exclusive GPU lease
- `lab.core.llm` — litellm chat wrapper
- `lab.core.minio_io` — MinIO/S3 helpers (run_key, upload_bytes)
- `lab.core.daily_log` — daily lab journal helpers
- `lab.migrations` — SQL migrations (kept under `lab/migrations/` for the CLI)

## Gotchas
- Must remain dependency-free of other `lab-*` packages.
- PEP 420 namespace: no `src/lab/__init__.py` in this package.
