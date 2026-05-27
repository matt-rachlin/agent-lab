"""Lightweight notification helper — ntfy.sh + local notify-send fallback.

Defaults to ntfy.sh with a per-user topic derived from the lab git remote /
hostname. Override via `LAB_NTFY_URL` (full URL) or `LAB_NTFY_TOPIC` (topic only).
If neither is configured, falls back to `notify-send` on Linux desktops.
"""

from __future__ import annotations

import hashlib
import os
import socket
import subprocess
from typing import Literal

import httpx

DEFAULT_NTFY_BASE = "https://ntfy.sh"


def _default_topic() -> str:
    """Stable, hard-to-guess topic derived from hostname + user.

    Ntfy topics are public-by-default (anyone with the URL can read), so we
    avoid using the literal hostname. SHA256 → first 16 hex chars is sufficient
    for "obscurity-as-privacy" on a small lab.
    """
    seed = f"{socket.gethostname()}::{os.environ.get('USER', 'unknown')}::lab"
    return "lab-" + hashlib.sha256(seed.encode()).hexdigest()[:16]


def get_ntfy_url() -> str | None:
    """Resolve the ntfy URL from env, or None if neither URL nor topic configured."""
    explicit = os.environ.get("LAB_NTFY_URL")
    if explicit:
        return explicit
    topic = os.environ.get("LAB_NTFY_TOPIC")
    if topic is None:
        topic = _default_topic()
    base = os.environ.get("LAB_NTFY_BASE", DEFAULT_NTFY_BASE).rstrip("/")
    return f"{base}/{topic}"


def notify(
    message: str,
    *,
    title: str | None = None,
    priority: Literal["min", "low", "default", "high", "max"] = "default",
    tags: list[str] | None = None,
    click: str | None = None,
) -> bool:
    """Send a notification via ntfy + best-effort local notify-send. Returns True on any success."""
    ok = False

    url = get_ntfy_url()
    if url:
        headers: dict[str, str] = {"Priority": priority}
        if title:
            headers["Title"] = title
        if tags:
            headers["Tags"] = ",".join(tags)
        if click:
            headers["Click"] = click
        try:
            r = httpx.post(url, content=message.encode("utf-8"), headers=headers, timeout=10)
            r.raise_for_status()
            ok = True
        except httpx.HTTPError:
            pass

    # Local desktop notification (Linux only, best-effort)
    try:
        cmd = ["notify-send", "-a", "lab"]
        if priority in ("high", "max"):
            cmd += ["-u", "critical"]
        if title:
            cmd += [title, message]
        else:
            cmd += [message]
        subprocess.run(
            cmd, check=False, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        ok = True
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    return ok
