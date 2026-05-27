"""pdf fetcher: download PDF, extract with pymupdf preserving headings.

Vendored from kb_builder.fetchers.pdf.
"""

from __future__ import annotations

import httpx

from lab.rag._plan import PlannedSource
from lab.rag.fetchers import FetchedDoc, FetcherContext, FetchResult, register


def _pdf_to_markdown(data: bytes) -> tuple[str, str | None]:
    import fitz  # pymupdf

    md_parts: list[str] = []
    title = None
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"pymupdf open failed: {e}") from e
    meta = doc.metadata or {}
    title = meta.get("title")
    # `fitz.Document` implements iteration via `__getitem__` (sequence
    # protocol), which pyright doesn't recognize as `Iterable`. Iterate by
    # index instead — equivalent at runtime, accepted by pyright.
    # `Page.get_text("text")` returns a `str` at runtime but pymupdf's
    # overloads also allow list/dict shapes for other modes; coerce to
    # str so the str-only operations below type-check cleanly.
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = str(page.get_text("text") or "")
        if text.strip():
            md_parts.append(f"\n\n## Page {i + 1}\n\n{text.strip()}\n")
    doc.close()
    return "".join(md_parts), title


def fetch(source: PlannedSource, ctx: FetcherContext) -> FetchResult:
    ctx.rate_limiter.wait(source.url)
    try:
        r = httpx.get(
            source.url, timeout=60.0, follow_redirects=True, headers={"User-Agent": ctx.user_agent}
        )
        r.raise_for_status()
    except Exception as e:
        return FetchResult(skipped=True, skipped_reason=f"pdf fetch error: {e}")
    try:
        md, title = _pdf_to_markdown(r.content)
    except Exception as e:
        return FetchResult(skipped=True, skipped_reason=f"pdf extract error: {e}")
    if len(md.strip()) < 200:
        return FetchResult(skipped=True, skipped_reason="empty PDF extraction")
    doc = FetchedDoc(
        url=str(r.url),
        raw_bytes=r.content,
        raw_ext=".pdf",
        body_markdown=md,
        title=title,
        license=source.license,
    )
    ctx.budget.add_page()
    return FetchResult(docs=[doc])


register("pdf", fetch)
