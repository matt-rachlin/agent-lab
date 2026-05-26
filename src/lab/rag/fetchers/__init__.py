"""Source-type-aware fetcher dispatcher.

Vendored from kb_builder.fetchers. Lab's 6h-a CLI does not invoke fetchers,
but the modules stay importable so a later sub-phase (or downstream user) can
drive KB construction through `lab.rag.fetchers.get(name)(planned, ctx)`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab.rag._budget import BudgetTracker, DomainRateLimiter
from lab.rag._plan import PlannedSource


@dataclass
class FetchResult:
    """One or more documents extracted from a single planned source.

    Many sources are single-doc (e.g. one HTML page). Some (git-repo, sitemap)
    fan out to many documents.
    """

    docs: list[FetchedDoc] = field(default_factory=list)
    skipped: bool = False
    skipped_reason: str | None = None


@dataclass
class FetchedDoc:
    url: str  # final URL (post-redirect, or man:NAME(N), or git path)
    raw_bytes: bytes
    raw_ext: str  # ".html", ".pdf", ".txt", ".md", ".1"
    body_markdown: str  # extracted markdown
    title: str | None = None
    license: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


FetcherFn = Callable[[PlannedSource, "FetcherContext"], FetchResult]


@dataclass
class FetcherContext:
    kb_dir: Path
    budget: BudgetTracker
    rate_limiter: DomainRateLimiter
    user_agent: str = "lab.rag/0.1 (+https://github.com/local)"


_REGISTRY: dict[str, FetcherFn] = {}


def register(name: str, fn: FetcherFn) -> None:
    _REGISTRY[name] = fn


def get(name: str) -> FetcherFn:
    if name not in _REGISTRY:
        raise KeyError(f"no fetcher registered for {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


# Eager registration via imports
from lab.rag.fetchers import (  # noqa: F401, E402
    git,
    html_single,
    html_sitemap,
    html_spa,
    manpage,
    pdf,
    rfc,
    stack_exchange,
    webfetch,
)
