"""stack-exchange fetcher: minimal Stack Exchange API integration.

Vendored from kb_builder.fetchers.stack_exchange.
v1: opt-in only. Disabled by default (skipped with reason).
"""

from __future__ import annotations

import os

from lab.rag._plan import PlannedSource
from lab.rag.fetchers import FetcherContext, FetchResult, register


def fetch(source: PlannedSource, ctx: FetcherContext) -> FetchResult:
    if not os.environ.get("KB_STACKEXCHANGE_ENABLE"):
        return FetchResult(
            skipped=True,
            skipped_reason="stack-exchange disabled (set KB_STACKEXCHANGE_ENABLE=1 to opt in)",
        )
    # If enabled, defer to html_single on the URL — SO answers render fine that way.
    from lab.rag.fetchers.html_single import fetch as fetch_single

    return fetch_single(source, ctx)


register("stack_exchange", fetch)
