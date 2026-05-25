"""Daily log scaffolding — open or create docs/log/YYYY-MM-DD.md.

If yesterday's log exists and has a `## Tomorrow` section with items, those
items get pre-pasted into today's `## Intent today` section so the daily handoff
is one less step.
"""

from __future__ import annotations

import os
import re
import subprocess
from datetime import date, timedelta
from pathlib import Path

LOG_DIR_DEFAULT = Path("/data/lab/code/docs/log")

_TEMPLATE = """# {date}

## Intent today
{intent}
## Did
-

## Stuck on / questions
-

## Notes

## Tomorrow
-
"""


def _yesterdays_tomorrow(today: date, log_dir: Path) -> list[str]:
    """Return the bulleted items under yesterday's `## Tomorrow` section, if any."""
    yesterday = (today - timedelta(days=1)).isoformat()
    yf = log_dir / f"{yesterday}.md"
    if not yf.is_file():
        return []
    text = yf.read_text(encoding="utf-8")
    # find `## Tomorrow` block until next heading or EOF
    m = re.search(r"^##\s+Tomorrow\s*$(?P<body>.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    if not m:
        return []
    body = m.group("body")
    items = [line.strip() for line in body.splitlines() if line.strip().startswith("- ")]
    return [it for it in items if it not in ("- ",)]


def ensure_today(
    *, log_dir: Path = LOG_DIR_DEFAULT, today: date | None = None
) -> tuple[Path, bool]:
    """Return (today_path, created_flag). Pre-pasted intent if yesterday had Tomorrow items."""
    today = today or date.today()
    log_dir.mkdir(parents=True, exist_ok=True)
    target = log_dir / f"{today.isoformat()}.md"
    if target.exists():
        return target, False
    pre = _yesterdays_tomorrow(today, log_dir)
    intent = "Carried from yesterday:\n" + "\n".join(pre) + "\n\n" if pre else "- \n\n"
    target.write_text(_TEMPLATE.format(date=today.isoformat(), intent=intent), encoding="utf-8")
    return target, True


def open_in_editor(path: Path) -> int:
    """Spawn $EDITOR (fallback to $VISUAL, then `vi`) on the path. Returns exit code."""
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    return subprocess.call([editor, str(path)])
