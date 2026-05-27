"""Integration: Valkey-backed GPU lease acquire/release round-trip."""

from __future__ import annotations

from typing import Any

import pytest
from lab.core.gpu_lease import LEASE_KEY, force_release, gpu_lease, status

pytestmark = pytest.mark.integration


def test_lease_acquire_release(valkey: Any) -> None:
    """Round-trip the lease and verify Valkey state at each step."""
    # Start clean — best effort
    valkey.delete(LEASE_KEY)

    with gpu_lease("integration-test", ttl_sec=10) as tag:
        holder, ttl = status()
        assert holder == tag, f"expected our tag, got {holder!r}"
        assert 0 < ttl <= 10
    holder, _ = status()
    assert holder is None, f"lease should be released; still {holder!r}"


def test_force_release_clears(valkey: Any) -> None:
    valkey.delete(LEASE_KEY)
    valkey.set(LEASE_KEY, "stale-holder", ex=300)
    assert force_release() is True
    assert valkey.get(LEASE_KEY) is None
