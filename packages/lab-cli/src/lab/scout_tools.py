"""In-process scout tools (ADR-011): source-API search + general web search
(SearXNG) + SSRF-guarded fetch + cited-add. NOT the sandboxed lab.agent.tools
(MCP/podman). Plain callables + hand-written OpenAI tool schemas for the driver
loop.

v2 adds web_search: general-web discovery via the lab-local SearXNG JSON API.
SearXNG is a trusted local service (loopback), so web_search talks to it
directly — it deliberately bypasses the public-host guard that fetch_url uses.
The result URLs web_search returns are public and are still verified through the
SSRF-guarded fetch_url / scout_add path before anything is logged.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import subprocess
from typing import Any
from urllib.parse import quote, urlparse

import httpx
from lab.platform.agent_runtime import Tool

from lab.scout import add_recommendation

_UA = "Mozilla/5.0 (compatible; lab-scout/0.2)"
_REACHABLE = {200, 301, 302, 303, 307, 308}
_CATEGORIES = ("model", "architecture", "software", "paper", "method", "benchmark")
_CONFIDENCE = ("low", "medium", "high")
# lab-local SearXNG (compose service lab-searxng); loopback, trusted.
_SEARXNG_URL = os.environ.get("LAB_SEARXNG_URL", "http://localhost:8888").rstrip("/")
_SEARXNG_CATEGORIES = ("general", "science", "it", "news")


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


def web_search(
    query: str, max_results: int = 6, categories: str = "general"
) -> list[dict[str, Any]]:
    """General-web search via the lab-local SearXNG JSON API (blogs, news, docs,
    forums — beyond arXiv/GitHub). Talks to the trusted loopback service
    directly; returned URLs are still verified via fetch_url/scout_add."""
    if categories not in _SEARXNG_CATEGORIES:
        categories = "general"
    try:
        r = httpx.get(
            f"{_SEARXNG_URL}/search",
            params={"q": query, "format": "json", "categories": categories},
            headers={"User-Agent": _UA},
            timeout=20.0,
        )
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError) as exc:
        return [{"error": str(exc)}]
    out: list[dict[str, Any]] = []
    for item in data.get("results", [])[: int(max_results)]:
        out.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": (item.get("content") or "")[:400],
            }
        )
    return out


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


_INT: dict[str, str] = {"type": "integer"}


def build_tools() -> list[Tool]:
    """The scout's tools as ADR-012 Tool ABI instances (in-process backend).
    Search/fetch are external_read; scout_add mutates the rec store (write_local)."""
    return [
        Tool(
            name="web_search",
            description=(
                "General-web search (blogs, news, docs, forums) via SearXNG. Use this "
                "first for broad discovery; arxiv_search/github_search for papers/code. "
                "categories: general|science|it|news."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": _INT,
                    "categories": {"type": "string", "enum": list(_SEARXNG_CATEGORIES)},
                },
                "required": ["query"],
            },
            impl=web_search,
            side_effect="external_read",
        ),
        Tool(
            name="arxiv_search",
            description="Search arXiv for papers.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}, "max_results": _INT},
                "required": ["query"],
            },
            impl=arxiv_search,
            side_effect="external_read",
        ),
        Tool(
            name="github_search",
            description="Search GitHub repos.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}, "max_results": _INT},
                "required": ["query"],
            },
            impl=github_search,
            side_effect="external_read",
        ),
        Tool(
            name="fetch_url",
            description="Fetch + extract a public URL to verify/quote a source.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            impl=fetch_url,
            side_effect="external_read",
        ),
        Tool(
            name="scout_add",
            description="Log a cited recommendation (verifies reachability; deduped).",
            parameters={
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
            impl=scout_add_tool,
            side_effect="write_local",
        ),
    ]
