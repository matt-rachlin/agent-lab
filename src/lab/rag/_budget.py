"""Budget tracking + per-domain rate limiter.

Vendored from kb_builder.budget. Used by lab.rag.fetchers. Lab's 6h-a CLI does
not build KBs, but the fetchers' import surface must stay self-contained.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from urllib.parse import urlparse

from lab.rag.manifest import Budget


class BudgetExceeded(RuntimeError):
    pass


class BudgetTracker:
    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.pages = 0
        self.tokens_embedded = 0
        self.start_ts = time.monotonic()
        self._lock = threading.Lock()

    def add_page(self) -> None:
        with self._lock:
            self.pages += 1
            if self.pages > self.budget.max_pages:
                raise BudgetExceeded(f"page budget exceeded (>{self.budget.max_pages})")

    def add_tokens(self, n: int) -> None:
        with self._lock:
            self.tokens_embedded += n
            if self.tokens_embedded > self.budget.max_tokens_embedded:
                raise BudgetExceeded(f"token budget exceeded (>{self.budget.max_tokens_embedded})")

    def check_time(self) -> None:
        elapsed_min = (time.monotonic() - self.start_ts) / 60.0
        if elapsed_min > self.budget.max_wall_minutes:
            raise BudgetExceeded(
                f"wall-time budget exceeded ({elapsed_min:.1f} > {self.budget.max_wall_minutes} min)"
            )


class DomainRateLimiter:
    """Min seconds between requests per domain."""

    def __init__(self, default_seconds: float = 1.0) -> None:
        self.default = default_seconds
        self._last: dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()

    def wait(self, url: str, seconds: float | None = None) -> None:
        host = urlparse(url).netloc or "_local"
        delay = self.default if seconds is None else seconds
        with self._lock:
            now = time.monotonic()
            ready = self._last[host] + delay
            sleep_for = ready - now if now < ready else 0.0
            self._last[host] = max(now, ready)
        if sleep_for > 0:
            time.sleep(sleep_for)
