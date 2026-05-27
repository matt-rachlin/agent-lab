"""rfc fetcher: pull plain-text RFC from rfc-editor.org.

Vendored from kb_builder.fetchers.rfc.
"""

from __future__ import annotations

import re

import httpx

from lab.rag._plan import PlannedSource
from lab.rag.fetchers import FetchedDoc, FetcherContext, FetchResult, register

_RFC_NUM_RE = re.compile(r"rfc(\d+)", re.I)


def _to_text_url(url: str) -> str:
    """Normalize an rfc URL to its txt form on rfc-editor.org."""
    m = _RFC_NUM_RE.search(url)
    if not m:
        return url
    num = m.group(1)
    return f"https://www.rfc-editor.org/rfc/rfc{num}.txt"


def fetch(source: PlannedSource, ctx: FetcherContext) -> FetchResult:
    url = _to_text_url(source.url)
    ctx.rate_limiter.wait(url)
    try:
        r = httpx.get(url, timeout=30.0, headers={"User-Agent": ctx.user_agent})
        r.raise_for_status()
    except Exception as e:
        return FetchResult(skipped=True, skipped_reason=f"rfc fetch error: {e}")
    text = r.text
    if len(text.strip()) < 500:
        return FetchResult(skipped=True, skipped_reason="rfc body suspiciously short")
    body = "```\n" + text + "\n```"
    title = None
    # Heuristic: title is typically near the top, after the from-line block
    for ln in text.splitlines()[:60]:
        sline = ln.strip()
        if sline and not sline.startswith(("Network Working", "Request for", "Category:", "ISSN")):
            title = sline
            break
    doc = FetchedDoc(
        url=url,
        raw_bytes=r.content,
        raw_ext=".txt",
        body_markdown=f"# {title or url}\n\n{body}\n",
        title=title or url,
        license="RFC-Trust",
    )
    ctx.budget.add_page()
    return FetchResult(docs=[doc])


register("rfc", fetch)
