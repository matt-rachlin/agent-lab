"""html-sitemap fetcher: walk a sitemap.xml, fetch + extract each URL.

Vendored from kb_builder.fetchers.html_sitemap.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

import httpx
from lab.rag._plan import PlannedSource
from lab.rag.fetchers import FetcherContext, FetchResult, register
from lab.rag.fetchers.html_single import fetch as fetch_single

SM_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


def _parse_sitemap(text: str) -> list[str]:
    urls: list[str] = []
    try:
        root = ET.fromstring(text)  # noqa: S314  # reason: sitemaps are public URL lists; no XXE surface
    except ET.ParseError:
        return urls
    # Sitemap index?
    if root.tag.endswith("sitemapindex"):
        for sm in root.findall(f"{SM_NS}sitemap/{SM_NS}loc"):
            urls.append(sm.text or "")
    else:
        for u in root.findall(f"{SM_NS}url/{SM_NS}loc"):
            if u.text:
                urls.append(u.text)
    return [u for u in urls if u]


def _candidate_sitemap_urls(url: str) -> list[str]:
    """Try the URL as-given, plus common conventional sitemap locations.

    If the agent listed e.g. https://example.com/docs/ as html-sitemap, also
    probe https://example.com/sitemap.xml and https://example.com/docs/sitemap.xml.
    """
    from urllib.parse import urlparse, urlunparse

    cands = [url]
    if not url.endswith(".xml"):
        # Try appending sitemap.xml under the listed path
        if url.endswith("/"):
            cands.append(url + "sitemap.xml")
        else:
            cands.append(url + "/sitemap.xml")
        # Try root /sitemap.xml
        p = urlparse(url)
        cands.append(urlunparse((p.scheme, p.netloc, "/sitemap.xml", "", "", "")))
    # de-dup, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _try_sitemap(url: str, ctx: FetcherContext) -> list[str]:
    """Return URL list from one sitemap candidate, or [] on failure."""
    ctx.rate_limiter.wait(url)
    try:
        r = httpx.get(url, timeout=30.0, follow_redirects=True)
        r.raise_for_status()
    except Exception:
        return []
    if "<urlset" not in r.text and "<sitemapindex" not in r.text:
        return []
    urls = _parse_sitemap(r.text)
    if urls and urls[0].endswith(".xml"):
        more: list[str] = []
        for u in urls[:3]:
            ctx.rate_limiter.wait(u)
            try:
                rr = httpx.get(u, timeout=30.0, follow_redirects=True)
                rr.raise_for_status()
                more.extend(_parse_sitemap(rr.text))
            except Exception:  # noqa: S110  # reason: best-effort sub-sitemap fetch; ignore individual failures
                pass
        urls = more
    return urls


def fetch(source: PlannedSource, ctx: FetcherContext) -> FetchResult:
    # Try the URL plus conventional sitemap locations
    urls: list[str] = []
    for cand in _candidate_sitemap_urls(source.url):
        urls = _try_sitemap(cand, ctx)
        if urls:
            break

    if not urls:
        # Fallback: treat the listed URL as a single page. This catches the
        # common agent misclassification of "wiki landing page" as sitemap.
        single = PlannedSource(
            url=source.url,
            type="html-single-page",
            authority=source.authority,
            inclusion_rationale=f"sitemap-fallback: {source.inclusion_rationale}",
        )
        return fetch_single(single, ctx)

    urls = urls[:200]
    out_docs = []
    for u in urls:
        sub = PlannedSource(
            url=u,
            type="html-single-page",
            authority=source.authority,
            inclusion_rationale=source.inclusion_rationale,
        )
        try:
            ctx.budget.check_time()
            res = fetch_single(sub, ctx)
        except Exception:  # noqa: S112  # reason: skip individual sitemap entries on error
            continue
        if res.docs:
            out_docs.extend(res.docs)
    if not out_docs:
        # Sitemap parsed but every URL extraction failed → still try the landing page
        single = PlannedSource(
            url=source.url,
            type="html-single-page",
            authority=source.authority,
            inclusion_rationale=f"sitemap-empty-fallback: {source.inclusion_rationale}",
        )
        return fetch_single(single, ctx)
    return FetchResult(docs=out_docs)


register("html_sitemap", fetch)
