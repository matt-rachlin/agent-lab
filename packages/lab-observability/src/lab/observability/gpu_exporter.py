"""Minimal Prometheus exporter for nvidia-smi metrics.

Lighter than NVIDIA DCGM — no daemon, no driver kit, just `nvidia-smi`
shelled out every scrape. Runs as a systemd user service (see scripts/).

Exposes :9400/metrics with a subset of DCGM_FI_* names so the same Grafana
dashboards that target DCGM mostly work.
"""

from __future__ import annotations

import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

QUERIES = [
    "index",
    "name",
    "uuid",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "power.draw",
    "power.limit",
    "clocks.current.sm",
    "clocks.current.memory",
    "fan.speed",
    "pcie.link.gen.current",
    "pstate",
]


def _query() -> list[dict[str, str]]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(QUERIES)}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows: list[dict[str, str]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(QUERIES):
            continue
        rows.append(dict(zip(QUERIES, parts, strict=True)))
    return rows


def _emit_metric(
    name: str, help_text: str, mtype: str, samples: list[tuple[dict[str, str], float]]
) -> str:
    lines = [f"# HELP {name} {help_text}", f"# TYPE {name} {mtype}"]
    for labels, value in samples:
        if not labels:
            lines.append(f"{name} {value}")
        else:
            label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
            lines.append(f"{name}{{{label_str}}} {value}")
    return "\n".join(lines) + "\n"


def _try_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def render_metrics() -> str:
    try:
        rows = _query()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return f"# nvidia-smi failed: {exc}\n"

    out: list[str] = []

    def emit(name: str, help_text: str, mtype: str, field: str, scale: float = 1.0) -> None:
        samples: list[tuple[dict[str, str], float]] = []
        for r in rows:
            v = _try_float(r.get(field, ""))
            if v is None:
                continue
            samples.append(
                (
                    {
                        "gpu": r.get("index", "?"),
                        "name": r.get("name", "?"),
                        "uuid": r.get("uuid", "?"),
                    },
                    v * scale,
                )
            )
        if samples:
            out.append(_emit_metric(name, help_text, mtype, samples))

    # Memory (DCGM compat names where reasonable)
    emit(
        "DCGM_FI_DEV_FB_TOTAL",
        "Framebuffer total bytes",
        "gauge",
        "memory.total",
        scale=1024 * 1024,
    )
    emit("DCGM_FI_DEV_FB_USED", "Framebuffer used bytes", "gauge", "memory.used", scale=1024 * 1024)
    emit("DCGM_FI_DEV_FB_FREE", "Framebuffer free bytes", "gauge", "memory.free", scale=1024 * 1024)
    # Utilization
    emit("DCGM_FI_DEV_GPU_UTIL", "GPU utilization %", "gauge", "utilization.gpu")
    emit("DCGM_FI_DEV_MEM_COPY_UTIL", "Memory copy utilization %", "gauge", "utilization.memory")
    emit("DCGM_FI_DEV_GPU_TEMP", "GPU temperature C", "gauge", "temperature.gpu")
    emit("DCGM_FI_DEV_POWER_USAGE", "Power draw W", "gauge", "power.draw")
    emit("nvidia_smi_power_limit_w", "Power limit W", "gauge", "power.limit")
    emit("DCGM_FI_DEV_SM_CLOCK", "SM clock MHz", "gauge", "clocks.current.sm")
    emit("DCGM_FI_DEV_MEM_CLOCK", "Memory clock MHz", "gauge", "clocks.current.memory")
    emit("nvidia_smi_fan_speed_pct", "Fan speed %", "gauge", "fan.speed")

    return "".join(out)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body = render_metrics().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        # quiet — we don't want per-scrape stdout spam
        return


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9400)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"lab gpu_exporter listening on http://{args.host}:{args.port}/metrics")
    server.serve_forever()


if __name__ == "__main__":
    main()
