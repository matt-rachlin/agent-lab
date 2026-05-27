"""webfetch fetcher: last-resort, single page via httpx + trafilatura.

Vendored from kb_builder.fetchers.webfetch. Identical to html_single but
documents an explicit intent: the agent picked this when it couldn't
classify the source.
"""

from __future__ import annotations

from lab.rag._plan import PlannedSource
from lab.rag.fetchers import FetcherContext, FetchResult, register
from lab.rag.fetchers.html_single import fetch as fetch_single


def fetch(source: PlannedSource, ctx: FetcherContext) -> FetchResult:
    return fetch_single(source, ctx)


register("webfetch", fetch)
