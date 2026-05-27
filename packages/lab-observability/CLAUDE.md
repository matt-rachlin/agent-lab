# lab-observability

Monitoring + cost guards:
- `lab.observability.gpu_exporter` — Prometheus exporter for nvidia-smi
- `lab.observability.spend` — running spend ledger
- `lab.observability.quota` — usage windows + alerts

## Gotchas
- gpu_exporter runs as a systemd user service; entrypoint is `python -m lab.observability.gpu_exporter`.
- spend/quota both read/write the Postgres `runs` table via `lab.core.settings`.
