"""In-process scout tools (ADR-011 v1): source-API search + SSRF-guarded fetch +
cited-add. NOT the sandboxed lab.agent.tools (MCP/podman) — that is v2. Plain
callables + hand-written OpenAI tool schemas for the driver loop.
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import subprocess
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from lab.scout import add_recommendation

_UA = "Mozilla/5.0 (compatible; lab-scout/0.1)"
_REACHABLE = {200, 301, 302, 303, 307, 308}
_CATEGORIES = ("model", "architecture", "software", "paper", "method", "benchmark")
_CONFIDENCE = ("low", "medium", "high")


def _is_public_host(host: str) -> bool:
    """SSRF guard: True only if every resolved IP is a public address."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return bool(infos)


def _safe_get(url: str, *, timeout: float = 15.0) -> httpx.Response:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname or not _is_public_host(p.hostname):
        raise ValueError(f"blocked url (non-public host or bad scheme): {url}")
    return httpx.get(url, follow_redirects=True, headers={"User-Agent": _UA}, timeout=timeout)


def fetch_url(url: str) -> dict[str, Any]:
    """Fetch a public URL (SSRF-guarded, follows redirects); return status +
    extracted text (trafilatura best-effort, capped)."""
    try:
        r = _safe_get(url)
    except (httpx.HTTPError, ValueError, OSError) as exc:
        return {"status": 0, "error": str(exc), "text": ""}
    text = r.text[:8000]
    try:
        import trafilatura

        extracted = trafilatura.extract(r.text)
        if extracted:
            text = extracted[:8000]
    except (ImportError, ValueError, TypeError):
        pass
    return {"status": r.status_code, "text": text}


def arxiv_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    url = (
        "http://export.arxiv.org/api/query?search_query=all:"
        + quote(query)
        + f"&start=0&max_results={int(max_results)}"
    )
    try:
        r = _safe_get(url, timeout=20)
    except (httpx.HTTPError, ValueError, OSError) as exc:
        return [{"error": str(exc)}]
    out: list[dict[str, Any]] = []
    for m in re.finditer(r"<entry>(.*?)</entry>", r.text, re.DOTALL):
        e = m.group(1)

        def _g(tag: str, blk: str = e) -> str:
            mm = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", blk, re.DOTALL)
            return re.sub(r"\s+", " ", mm.group(1)).strip() if mm else ""

        out.append({"title": _g("title"), "url": _g("id"), "summary": _g("summary")[:400]})
    return out[:max_results]


def github_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Fixed-arg `gh search repos` (no passthrough)."""
    try:
        proc = subprocess.run(
            [
                "gh",
                "search",
                "repos",
                "--json",
                "fullName,url,description",
                "--limit",
                str(int(max_results)),
                "--",
                query,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [{"error": str(exc)}]
    if proc.returncode != 0:
        return [{"error": proc.stderr.strip()[:200]}]
    try:
        result: list[dict[str, Any]] = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return result


def scout_add_tool(
    source_url: str, title: str, category: str, why: str, confidence: str = "medium"
) -> dict[str, Any]:
    """Cited-fetch-before-add: verify reachability, then log a deduped rec."""
    if category not in _CATEGORIES:
        return {"result": "error", "msg": f"category must be one of {list(_CATEGORIES)}"}
    if confidence not in _CONFIDENCE:
        return {"result": "error", "msg": "confidence must be low|medium|high"}
    blocked = False
    try:
        r = _safe_get(source_url, timeout=12)
        reachable = r.status_code in _REACHABLE
        blocked = r.status_code in (403, 429)
    except (httpx.HTTPError, ValueError, OSError):
        reachable = False
    if not reachable and not blocked:
        return {"result": "unreachable", "msg": f"source_url not reachable: {source_url}"}
    res = add_recommendation(
        source_url=source_url,
        title=title,
        category=category,
        why_relevant=why,
        confidence=confidence,
    )
    return {"result": res, "blocked_but_accepted": blocked}


# OpenAI tool schemas for the driver loop + dispatch map.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "arxiv_search",
            "description": "Search arXiv for papers.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "github_search",
            "description": "Search GitHub repos.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch + extract a public URL to verify/quote a source.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scout_add",
            "description": "Log a cited recommendation (verifies reachability; deduped).",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_url": {"type": "string"},
                    "title": {"type": "string"},
                    "category": {"type": "string", "enum": list(_CATEGORIES)},
                    "why": {"type": "string"},
                    "confidence": {"type": "string", "enum": list(_CONFIDENCE)},
                },
                "required": ["source_url", "title", "category", "why"],
            },
        },
    },
]

DISPATCH = {
    "arxiv_search": arxiv_search,
    "github_search": github_search,
    "fetch_url": fetch_url,
    "scout_add": scout_add_tool,
}
