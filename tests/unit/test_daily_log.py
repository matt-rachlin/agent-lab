"""Daily log scaffold tests."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from lab.core.daily_log import _yesterdays_tomorrow, ensure_today


def test_ensure_today_creates_file(tmp_path: Path) -> None:
    today = date(2026, 5, 25)
    p, created = ensure_today(log_dir=tmp_path, today=today)
    assert created
    assert p.name == "2026-05-25.md"
    assert p.exists()


def test_ensure_today_idempotent(tmp_path: Path) -> None:
    today = date(2026, 5, 25)
    p1, c1 = ensure_today(log_dir=tmp_path, today=today)
    p2, c2 = ensure_today(log_dir=tmp_path, today=today)
    assert p1 == p2
    assert c1 is True
    assert c2 is False


def test_yesterdays_tomorrow_prefill(tmp_path: Path) -> None:
    today = date(2026, 5, 26)
    yesterday = today - timedelta(days=1)
    (tmp_path / f"{yesterday.isoformat()}.md").write_text(
        "# 2026-05-25\n\n## Tomorrow\n- finish phase 3\n- write F-003\n",
        encoding="utf-8",
    )
    items = _yesterdays_tomorrow(today, tmp_path)
    assert items == ["- finish phase 3", "- write F-003"]
    p, _ = ensure_today(log_dir=tmp_path, today=today)
    assert "finish phase 3" in p.read_text(encoding="utf-8")


def test_yesterdays_tomorrow_empty(tmp_path: Path) -> None:
    today = date(2026, 5, 26)
    yesterday = today - timedelta(days=1)
    (tmp_path / f"{yesterday.isoformat()}.md").write_text(
        "# 2026-05-25\n\n## Tomorrow\n- \n\n## Notes\n",
        encoding="utf-8",
    )
    assert _yesterdays_tomorrow(today, tmp_path) == []
