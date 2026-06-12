"""Guard: every task shipping _http_fixtures/ must set LAB_HTTP_FIXTURE_DIR.

Without it the sandbox http_fetch tool ignores the fixtures and the
allowlisted reserved domains resolve to the live internet (Cloudflare 404
pages) — the F-005 EXP-002 incident, which recurred in pbs-agent-brutal-v0.1
(BRUTAL-BENCH-001, 2026-06-12) because the previous guard was pinned to
tasks/pbs-agent-v0.1 only. This version scans every suite, including ones
that don't exist yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
FIXTURE_DIR = "/workspace/_http_fixtures"


def _suite_yamls() -> list[Path]:
    return sorted(p for p in (REPO / "tasks").glob("*/*.yaml"))


@pytest.mark.parametrize("path", _suite_yamls(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_http_tasks_set_fixture_dir(path: Path) -> None:
    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict) or "tasks" not in doc:
        pytest.skip("not a task suite file")
    offenders: list[str] = []
    for task in doc["tasks"]:
        sandbox = task.get("sandbox") or {}
        workspace = sandbox.get("workspace_files") or {}
        if not any(k.startswith("_http_fixtures/") for k in workspace):
            continue
        env = sandbox.get("env") or {}
        if env.get("LAB_HTTP_FIXTURE_DIR") != FIXTURE_DIR:
            offenders.append(str(task.get("slug")))
    assert not offenders, (
        f"{path}: tasks ship _http_fixtures/ without "
        f"LAB_HTTP_FIXTURE_DIR={FIXTURE_DIR}: {offenders} "
        "(fetches will hit the live internet — see F-005 / BRUTAL-BENCH-001)"
    )
