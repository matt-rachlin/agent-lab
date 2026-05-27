---
doc_id: observability-tracing
title: Runbook — observability tracing (Tempo + OTel)
zone: lab
kind: runbook
status: active
owner: m
created: '2026-05-27'
last_updated: '2026-05-27'
last_verified: '2026-05-27'
tags:
- lab
- runbook
- runbooks
- observability
- tracing
---

# Runbook — observability tracing (Tempo + OTel)

Phase 16.2 ships OpenTelemetry spans over every sweep cell. Tempo
receives them on `localhost:4317` (OTLP gRPC), Grafana queries them on
`localhost:3001` (Explore → Tempo data source). This runbook covers
day-to-day operation, common queries, and the contract for adding new
spans.

## Quick check

Tempo and Grafana are part of the lab compose stack:

```bash
podman ps --filter label=app=lab --format '{{.Names}}\t{{.Status}}'
```

Expect `lab-tempo`, `lab-grafana`, `lab-prometheus`, `lab-mlflow`,
`lab-minio`, `lab-litellm` all `Up`. If Tempo is missing:

```bash
podman-compose -f /data/lab/services/compose.yml up -d tempo
```

Verify Tempo is healthy:

```bash
curl -s http://localhost:3200/ready   # expect "ready"
curl -s http://localhost:3200/status  # service table
```

## Sending a test span

The simplest end-to-end check is to fire a span from a one-liner:

```bash
cd /data/lab/code && uv run python -c "
from lab.observability.tracing import configure_tracing, span
configure_tracing()
with span('test_root', **{'lab.test': 'smoke'}):
    with span('test_child', **{'lab.run_id': 'r-smoke'}):
        pass
from opentelemetry import trace
trace.get_tracer_provider().force_flush(5000)
"
```

Then search Tempo:

```bash
curl -s 'http://localhost:3200/api/search?tags=service.name%3Dlab&limit=5' | jq .
```

You should see a `rootTraceName: test_root` entry within a few seconds.

## Span hierarchy (sweep)

Every sweep cell produces this tree (single-turn fast path on the
left, multi-turn agent path on the right):

```
sweep_cell                          sweep_cell
├── manifest_capture                ├── manifest_capture
├── gpu_lease_acquire (local only)  ├── gpu_lease_acquire (local only)
├── litellm_call                    ├── inspect_eval
└── persist                         │   ├── agent_turn (×N)
                                    │   │   ├── litellm_call
                                    │   │   └── tool_call (×M, via ToolPool)
                                    │   └── ...
                                    └── persist
                                        ├── logwriter.upload_trajectory
                                        ├── logwriter.upsert_postgres
                                        └── logwriter.mlflow_mirror
```

Standard span attributes:

| attribute             | scope                | meaning                                      |
|-----------------------|----------------------|----------------------------------------------|
| `lab.run_id`          | sweep_cell + descendants | cell run_id (matches `experiment_runs.run_id`) |
| `lab.experiment_slug` | sweep_cell + descendants | experiment slug (e.g. EXP-004)               |
| `lab.model`           | sweep_cell + descendants | LiteLLM model id                             |
| `lab.task`            | sweep_cell           | task slug                                    |
| `lab.seed`            | sweep_cell           | seed integer                                 |
| `lab.config_hash`     | sweep_cell           | RunConfig hash                               |
| `lab.path`            | sweep_cell           | `single_turn` or `agent`                     |
| `lab.status`          | sweep_cell           | `done` / `error`                             |
| `lab.latency_ms`      | sweep_cell, litellm_call | wall-clock latency                       |
| `lab.tokens_in`       | sweep_cell, agent_turn | prompt tokens                              |
| `lab.tokens_out`      | sweep_cell, agent_turn | completion tokens                          |
| `lab.turn`            | agent_turn           | 0-indexed turn number                        |
| `tool.name`           | tool_call            | MCP tool name                                |
| `tool.module`         | tool_call            | dotted MCP server module                     |
| `tool.latency_ms`     | tool_call            | tool invocation latency                      |
| `tool.error`          | tool_call            | error string (only on failure)               |
| `error.type`          | any (errored)        | exception class name                         |
| `error.message`       | any (errored)        | exception repr                               |

## Common queries

Open Grafana → Explore → Tempo data source.

**One specific cell**

```
{ resource.service.name = "lab" && lab.run_id = "<paste run_id>" }
```

**Slowest sweep cells of the last hour**

```
{ resource.service.name = "lab" && name = "sweep_cell" } | duration > 30s
```

**All errored cells**

```
{ resource.service.name = "lab" && name = "sweep_cell" && status = error }
```

**LiteLLM call latency distribution per model**

```
{ resource.service.name = "lab" && name = "litellm_call" } | by(lab.model)
```

**Tool calls for one experiment**

```
{ resource.service.name = "lab" && lab.experiment_slug = "EXP-004" && name = "tool_call" }
```

## Adding a new span

Wherever you would have written a `print` / `console.log`, prefer:

```python
from lab.observability.tracing import span, current_span_attrs
from lab.observability.log import get_logger

log = get_logger(__name__)

with span("my_step", **{"lab.something": "value"}):
    # ... work ...
    current_span_attrs(**{"lab.result": 7})
    log.info("my_step_done", result=7)
```

Rules:

* Span names use snake_case (no `lab.` prefix in the name).
* Attribute keys use dotted lowercase. Prefix lab-specific keys with
  `lab.`, OTel-standard or domain prefixes (`tool.`, `error.`,
  `http.`) for the rest.
* Attribute values must be str / int / float / bool. `None` values are
  silently dropped; non-scalar values are stringified.
* On error, `span` automatically tags `error.type` + `error.message`
  and re-raises, so callers do not have to catch-and-tag manually.

## Sampling

`LAB_OTEL_SAMPLE_RATIO` (env var, default `1.0`) clamps to `[0, 1]`.
At the lab's volumes (≤ hundreds of cells per sweep) we sample 100%.
If a future bulk sweep would overwhelm Tempo, drop it to `0.1` (10%);
parent-based sampling keeps related spans together.

## Disabling export

Set `LAB_OTEL_EXPORTER_URL=none` to keep span creation in-process
without shipping anything. Useful for unit tests and one-off scripts
that don't have Tempo running.

## Configuration knobs

| Env var                   | Default                  | Meaning                                          |
|---------------------------|--------------------------|--------------------------------------------------|
| `LAB_OTEL_EXPORTER_URL`   | `http://localhost:4317`  | OTLP gRPC endpoint. `none` disables export.      |
| `LAB_OTEL_SAMPLE_RATIO`   | `1.0`                    | Trace sample ratio in `[0, 1]`.                  |
| `LAB_LOG_LEVEL`           | `INFO`                   | stdlib log level for the structured logger.      |
| `LAB_LOG_JSON`            | unset (auto)             | Force JSON or console mode; default auto-detects from TTY. |

## Related

* `lab.observability.log` — structlog wrapper with `bind_run_context`.
* `lab.observability.tracing` — OTel span helpers.
* `/data/lab/services/tempo.yaml` — Tempo monolithic config.
* `/data/lab/services/grafana-provisioning/datasources/lab.yml` —
  Grafana data source provisioning.
