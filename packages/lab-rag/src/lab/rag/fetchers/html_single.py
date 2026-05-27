"""html-single-page fetcher: httpx + trafilatura.

Vendored from kb_builder.fetchers.html_single.
"""

from __future__ import annotations

import httpx
import trafilatura

from lab.rag._plan import PlannedSource
from lab.rag.fetchers import FetchedDoc, FetcherContext, FetchResult, register


def _extract(html: str, url: str) -> tuple[str, str | None]:
    md = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        include_links=True,
        favor_recall=True,
    )
    if not md:
        # Last-ditch: take any text
        md = trafilatura.extract(html, url=url, output_format="txt") or ""
    meta = trafilatura.extract_metadata(html, default_url=url)
    title = meta.title if meta else None
    return md, title


def fetch(source: PlannedSource, ctx: FetcherContext) -> FetchResult:
    url = source.url
    ctx.rate_limiter.wait(url)
    try:
        r = httpx.get(
            url,
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": ctx.user_agent, "Accept": "text/html,application/xhtml+xml"},
        )
        r.raise_for_status()
    except Exception as e:
        return FetchResult(skipped=True, skipped_reason=f"http error: {e}")

    if not r.text.strip():
        return FetchResult(skipped=True, skipped_reason="empty body")

    md, title = _extract(r.text, url)
    if not md or len(md.strip()) < 200:
        return FetchResult(skipped=True, skipped_reason="extraction yielded ≤200 chars")

    doc = FetchedDoc(
        url=str(r.url),
        raw_bytes=r.text.encode("utf-8", errors="replace"),
        raw_ext=".html",
        body_markdown=md,
        title=title,
        license=source.license,
    )
    ctx.budget.add_page()
    return FetchResult(docs=[doc])


register("html_single", fetch)
