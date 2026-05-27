"""Valkey-backed GPU lease.

A single 12 GB GPU can't serve two big models at once. This is the bouncer:
acquire a lease before loading; release on exit. Lease has a TTL so a crashed
process can't deadlock the GPU forever.

Usage:

    with gpu_lease("qwen3:14b-q4", ttl_sec=3600):
        # ... do GPU work ...
"""

from __future__ import annotations

import os
import socket
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import redis

from lab.core.settings import get_settings

LEASE_KEY = "lab:gpu:lease"


class LeaseTimeout(RuntimeError):
    """Raised when the GPU lease could not be acquired in the given window."""


def _client() -> redis.Redis[str]:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def _holder_tag(owner: str) -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{owner}:{uuid.uuid4().hex[:8]}"


@contextmanager
def gpu_lease(
    owner: str,
    *,
    ttl_sec: int = 1800,
    wait_sec: float = 600.0,
    poll_sec: float = 1.0,
) -> Iterator[str]:
    """Acquire the GPU lease for the duration of the context.

    Args:
        owner: human label for the lease holder (model name, sweep id, etc.).
        ttl_sec: how long the lease lives if not renewed; auto-released on context exit.
        wait_sec: max time to wait for an existing lease to free up.
        poll_sec: how often to retry while waiting.

    Yields:
        the unique holder tag.

    Raises:
        LeaseTimeout if we can't acquire within `wait_sec`.
    """
    r = _client()
    tag = _holder_tag(owner)
    deadline = time.monotonic() + wait_sec

    while True:
        ok = r.set(LEASE_KEY, tag, nx=True, ex=ttl_sec)
        if ok:
            break
        if time.monotonic() >= deadline:
            current = r.get(LEASE_KEY) or "<expired>"
            raise LeaseTimeout(f"could not acquire GPU lease in {wait_sec:.0f}s; held by {current}")
        time.sleep(poll_sec)

    try:
        yield tag
    finally:
        # Release only if we still own it (don't release another holder's lease)
        script = (
            "if redis.call('GET', KEYS[1]) == ARGV[1] then "
            "return redis.call('DEL', KEYS[1]) else return 0 end"
        )
        r.eval(script, 1, LEASE_KEY, tag)  # type: ignore[no-untyped-call]


def status() -> tuple[str | None, int]:
    """Return (current holder tag or None, TTL seconds remaining or -1)."""
    r = _client()
    holder = r.get(LEASE_KEY)
    ttl = r.ttl(LEASE_KEY) if holder else -1
    return holder, ttl


def force_release() -> bool:
    """Forcibly clear the lease. Use only when you're sure no one is using it."""
    r = _client()
    return bool(r.delete(LEASE_KEY))


def main() -> None:
    """`uv run python -m lab.gpu_lease --test` — round-trip a lease."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()

    if args.release:
        released = force_release()
        print(f"force release: {'cleared' if released else 'was already free'}")
        return
    if args.status:
        holder, ttl = status()
        print(f"holder: {holder}  ttl: {ttl}s")
        return
    if args.test:
        with gpu_lease("self-test", ttl_sec=30) as tag:
            holder, ttl = status()
            assert holder == tag, f"expected {tag}, got {holder}"
            print(f"acquired lease: {tag} (ttl {ttl}s)")
        holder, _ = status()
        assert holder is None, f"expected released, still held by {holder}"
        print("released OK")


if __name__ == "__main__":
    main()
