"""html-spa fetcher: render with playwright, then extract with trafilatura.

Vendored from kb_builder.fetchers.html_spa. Lazy-imports playwright so the
dep is optional.
"""

from __future__ import annotations

import trafilatura

from lab.rag._plan import PlannedSource
from lab.rag.fetchers import FetchedDoc, FetcherContext, FetchResult, register


def fetch(source: PlannedSource, ctx: FetcherContext) -> FetchResult:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return FetchResult(
            skipped=True,
            skipped_reason="playwright not installed (`pip install playwright && playwright install chromium`)",
        )

    ctx.rate_limiter.wait(source.url)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent=ctx.user_agent)
                page.goto(source.url, timeout=30000, wait_until="networkidle")
                html = page.content()
            finally:
                browser.close()
    except Exception as e:
        return FetchResult(skipped=True, skipped_reason=f"playwright error: {e}")

    md = trafilatura.extract(
        html, url=source.url, output_format="markdown", include_tables=True, favor_recall=True
    )
    if not md or len(md.strip()) < 200:
        return FetchResult(skipped=True, skipped_reason="SPA extraction yielded ≤200 chars")

    meta = trafilatura.extract_metadata(html, default_url=source.url)
    title = meta.title if meta else None
    doc = FetchedDoc(
        url=source.url,
        raw_bytes=html.encode("utf-8", errors="replace"),
        raw_ext=".html",
        body_markdown=md,
        title=title,
        license=source.license,
    )
    ctx.budget.add_page()
    return FetchResult(docs=[doc])


register("html_spa", fetch)
