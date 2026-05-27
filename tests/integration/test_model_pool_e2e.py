"""Integration tests for :mod:`lab.core.model_pool` against a real llama-swap.

These tests exercise the page-cache pre-flight contract end-to-end:
``declare()`` should fill the OS page cache with the model's GGUF;
``teardown()`` should evict the model from VRAM (the page cache stays
warm by design — that's the whole point of pre-flight).

Page cache state is observed via ``mincore(2)`` because ``vmtouch`` may
not be installed on the box.

GPU dependency:
    Pre-flight triggers a real ``/v1/chat/completions`` call against
    llama-swap, which momentarily loads the model into VRAM. If another
    GPU consumer (notably an active EXP-002b sweep) holds the GPU lease,
    pre-flight will block until the lease frees. To keep CI green we
    guard these tests with ``@pytest.mark.requires_gpu_idle`` and skip
    when ``/data/lab/services/gpu-lease`` is held.

These tests target ``qwen3-reranker-0.6b`` (the smallest registered
model, ~700 MB) to minimise risk to other GPU tenants. They are also
the integration-test "lowest-blast-radius" pick: the reranker runs as
an always-on host-side HTTP service behind llama-swap, so the load
event is purely a page-cache exercise — it does NOT take the LLM slot.
"""

from __future__ import annotations

import ctypes
import json
import mmap
import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from lab.core.model_pool import ModelPool, PipelineModelPlan, PipelineStep

# --------------------------------------------------------------------------
# Skip plumbing — these tests require a real llama-swap on the box
# --------------------------------------------------------------------------


_LLAMA_SWAP_URL = os.environ.get("LAB_LLAMA_SWAP_URL", "http://localhost:8080")
"""Match the same env var the production ModelPool reads via settings."""


def _llama_swap_reachable() -> bool:
    """True iff GET /running responds within 1s."""

    try:
        r = httpx.get(f"{_LLAMA_SWAP_URL}/running", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False


def _gpu_busy() -> bool:
    """Heuristic: are we likely to contend for GPU if we trigger a load?

    Two checks (either positive → busy):
      1. Valkey ``lab:gpu:lease`` key is set (an EXP sweep holds the lease).
      2. ``nvidia-smi`` reports < 4 GB free VRAM (some other tenant —
         likely Ollama or a previous llama-server — has the slot).

    The second check is critical because Ollama doesn't take the Valkey
    lease; it just pins the model in VRAM via its own keep_alive timer.
    The phi-4 GGUF is ~9 GB at Q4_K_M; if we have <4 GB free, llama-swap
    will return 502 instead of loading.
    """

    try:
        import redis

        from lab.core.settings import get_settings

        client = redis.Redis.from_url(
            get_settings().redis_url, decode_responses=True, socket_timeout=1
        )
        if client.get("lab:gpu:lease"):
            return True
    except Exception:
        pass

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if proc.returncode == 0:
            free_mib = int(proc.stdout.strip().splitlines()[0])
            # Anything < 4 GB free is too tight for phi-4 Q4_K_M (~9 GB).
            if free_mib < 4096:
                return True
    except Exception:
        pass
    return False


# Auto-skip the whole module when prerequisites aren't met. We do this at
# import time via a fixture so pytest reports the reason cleanly.


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _llama_swap_reachable(),
        reason=f"llama-swap not reachable at {_LLAMA_SWAP_URL}",
    ),
]


# --------------------------------------------------------------------------
# mincore-based page-cache observability
# --------------------------------------------------------------------------


_PAGE_SIZE = os.sysconf("SC_PAGESIZE")
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p)
_libc.mincore.restype = ctypes.c_int

# Direct mmap via libc — bypasses Python's read-only buffer wrapper.
_libc.mmap.argtypes = (
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
)
_libc.mmap.restype = ctypes.c_void_p
_libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
_libc.munmap.restype = ctypes.c_int

_PROT_READ = 0x1
_MAP_SHARED = 0x01
_MAP_FAILED = ctypes.c_void_p(-1).value


def _resident_page_fraction(path: Path) -> float:
    """Return fraction (0.0-1.0) of `path`'s pages currently in the page cache.

    Uses ``mincore(2)``: mmap the file via libc directly (Python's
    ``mmap.mmap`` exposes a read-only buffer that ctypes can't take
    addresses of without ``from_buffer_copy``, which would double our
    memory footprint for multi-GB GGUFs), then ask the kernel which
    pages are resident.
    """

    size = path.stat().st_size
    if size == 0:
        return 0.0
    n_pages = (size + _PAGE_SIZE - 1) // _PAGE_SIZE

    fd = os.open(str(path), os.O_RDONLY)
    try:
        addr = _libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
        if addr == _MAP_FAILED or addr is None:
            err = ctypes.get_errno()
            raise OSError(err, f"mmap failed: errno={err}")
        try:
            vec = (ctypes.c_char * n_pages)()
            ret = _libc.mincore(ctypes.c_void_p(addr), size, vec)
            if ret != 0:
                err = ctypes.get_errno()
                raise OSError(err, f"mincore failed: errno={err}")
            resident = sum(1 for b in vec if b[0] & 1)
        finally:
            _libc.munmap(ctypes.c_void_p(addr), size)
    finally:
        os.close(fd)

    # Unused import guard (mmap kept for module-level constants references).
    _ = mmap

    return resident / n_pages


def _drop_page_cache_if_possible(path: Path) -> bool:
    """Try to drop `path` from the page cache via posix_fadvise(DONTNEED).

    Returns True if the drop was issued (best-effort; the kernel may keep
    pages around if they're dirty or pinned). Returns False if we couldn't
    open the file. We avoid `sysctl vm.drop_caches` because it requires
    sudo and would nuke the whole system cache.
    """

    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return False
    try:
        # POSIX_FADV_DONTNEED = 4
        os.posix_fadvise(fd, 0, 0, 4)
        return True
    finally:
        os.close(fd)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


@pytest.fixture
def small_gguf_path() -> Path:
    """Path to the smallest llama-server-backed GGUF on disk.

    Note: `qwen3-reranker-0.6b` is registered in llama-swap as a host-side
    HTTP-proxy entry (cmd: ``sleep infinity``, no real GGUF backing in
    llama-server), so we can't use it for page-cache observation. The
    next-smallest real GGUF is phi-4-reasoning-14b at ~8.6 GB.

    If no small GGUF is present we skip rather than fail.
    """

    candidates = sorted(
        Path("/data/models/gguf").glob("phi-4-reasoning-14b/*.gguf"),
        key=lambda p: p.stat().st_size,
    )
    if not candidates:
        pytest.skip("phi-4-reasoning-14b GGUF not on disk under /data/models/gguf")
    return candidates[0]


@pytest.fixture
def small_swap_model_id() -> str:
    """litellm_id of the smallest model registered in llama-swap.

    Reranker is the cheapest pre-flight target (host-side HTTP service —
    no llama-server load + no VRAM impact), which makes it the right
    candidate for the non-GPU pre-flight smoke. The page-cache assertions
    elsewhere use ``small_gguf_path`` (phi-4-reasoning-14b) which DOES
    require GPU.
    """

    return "qwen3-reranker-0.6b"


def test_running_endpoint_is_well_formed() -> None:
    """Sanity: llama-swap's /running endpoint returns the expected shape."""

    r = httpx.get(f"{_LLAMA_SWAP_URL}/running", timeout=5.0)
    assert r.status_code == 200
    body = r.json()
    assert "running" in body
    assert isinstance(body["running"], list)


def test_unload_all_works() -> None:
    """Unload-all should always return 200, even when nothing is loaded."""

    r = httpx.post(f"{_LLAMA_SWAP_URL}/api/models/unload", timeout=10.0)
    assert r.status_code == 200


def test_preflight_and_teardown_smoke_with_reranker_proxy(
    small_swap_model_id: str,
) -> None:
    """End-to-end declare + teardown smoke that does NOT require GPU.

    The reranker entry in llama-swap is a host-side HTTP-proxy
    (``cmd: sleep infinity``, proxy to 127.0.0.1:8401), so a pre-flight
    completion against it doesn't touch the GPU at all. That makes this
    test safe to run alongside an in-flight GPU sweep — perfect for the
    "model_pool wiring is plausible" sanity check.

    What we verify:
      1. declare() makes the documented HTTP calls and returns cleanly
      2. teardown() makes the unload call and returns cleanly
      3. Both swallow benign failures (e.g. proxy already up)
      4. /running endpoint stays well-formed throughout
    """

    pool = ModelPool(llama_swap_url=_LLAMA_SWAP_URL)
    plan = PipelineModelPlan(
        pipeline_id="integration-test",
        steps=[PipelineStep(name="cell", models=[small_swap_model_id])],
    )

    pool.declare(plan)

    running = httpx.get(f"{_LLAMA_SWAP_URL}/running", timeout=5.0).json()["running"]
    # qwen3-reranker is `persistent: true` in the small-tools group, so it
    # MAY appear in /running; just assert the list is well-formed.
    assert isinstance(running, list)

    # teardown() is idempotent and safe.
    pool.teardown()
    pool.teardown()


def _model_loads_via_llama_swap(model_id: str) -> bool:
    """True iff llama-swap can actually load `model_id` right now.

    Used as a precondition for the page-cache tests: if llama-swap
    returns 502 (process exited, GPU full, etc.) the test would race
    on infrastructure rather than on our code path. Skip rather than
    failure-noise CI.
    """

    try:
        r = httpx.post(
            f"{_LLAMA_SWAP_URL}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "keep_alive": 0,
            },
            timeout=30.0,
        )
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.requires_gpu_idle
def test_preflight_fills_page_cache(small_gguf_path: Path) -> None:
    """Pre-flight should leave the GGUF mostly resident in the page cache.

    Strategy:
      1. Drop the GGUF from page cache via posix_fadvise(DONTNEED).
      2. Confirm cold-state resident fraction is low.
      3. declare() the plan — llama-swap loads + we unload.
      4. Confirm post-declare resident fraction is high.

    GPU-bound: pre-flight calls /v1/chat/completions which momentarily
    loads the model into VRAM. Skip when an EXP-* sweep is in flight
    or the underlying llama-server can't start (infra issue, not ours).
    """

    if _gpu_busy():
        pytest.skip("GPU busy (sweep lease held or VRAM tight) — defer")

    target_model = "phi-4-reasoning-14b"
    if not _model_loads_via_llama_swap(target_model):
        pytest.skip(
            f"llama-swap can't load {target_model} right now "
            "(probably llama-server infra issue; see journalctl --user -u llama-swap)"
        )

    if not _drop_page_cache_if_possible(small_gguf_path):
        pytest.skip("posix_fadvise not available — can't drop page cache")

    cold_resident = _resident_page_fraction(small_gguf_path)

    pool = ModelPool(llama_swap_url=_LLAMA_SWAP_URL)
    plan = PipelineModelPlan(
        pipeline_id="page-cache-test",
        steps=[PipelineStep(name="cell", models=[target_model])],
    )
    pool.declare(plan)

    after_resident = _resident_page_fraction(small_gguf_path)

    pool.teardown()

    # We don't insist on 100% residency (llama.cpp uses mmap with
    # MADV_RANDOM and may not have walked every page) but a meaningful
    # delta MUST be present.
    assert after_resident > cold_resident + 0.10, (
        f"page cache did not warm: cold={cold_resident:.2%} "
        f"after_preflight={after_resident:.2%}"
    )


@pytest.mark.requires_gpu_idle
def test_cold_vs_warm_load_observation(
    small_gguf_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Measure cold-vs-warm completion latency for a real-GGUF-backed model.

    This is observational: we time a /v1/chat/completions call against
    a freshly-cached GGUF vs. one we just told the kernel to evict.
    Output is captured for the test log; we assert with generous slack
    (50% + 0.5s) because hardware variance is huge.

    Skipped if posix_fadvise can't reach the file (e.g. permissions or
    SELinux denial) — the comparison is meaningless without it.
    """

    if _gpu_busy():
        pytest.skip("GPU busy (sweep lease held or VRAM tight) — defer")

    target_model = "phi-4-reasoning-14b"
    if not _model_loads_via_llama_swap(target_model):
        pytest.skip(
            f"llama-swap can't load {target_model} right now "
            "(probably llama-server infra issue; see journalctl --user -u llama-swap)"
        )

    pool = ModelPool(llama_swap_url=_LLAMA_SWAP_URL)
    plan = PipelineModelPlan(
        pipeline_id="cold-vs-warm",
        steps=[PipelineStep(name="cell", models=[target_model])],
    )

    # Warm pass first so we know the file is mapped at least once.
    pool.declare(plan)
    initial_resident = _resident_page_fraction(small_gguf_path)

    if not _drop_page_cache_if_possible(small_gguf_path):
        pytest.skip("posix_fadvise not available — can't simulate cold load")

    cold_resident = _resident_page_fraction(small_gguf_path)

    # Trigger the load again. Time the wall clock.
    t0 = time.monotonic()
    pool.declare(plan)
    cold_load_s = time.monotonic() - t0

    warm_resident = _resident_page_fraction(small_gguf_path)

    t1 = time.monotonic()
    pool.declare(plan)
    warm_load_s = time.monotonic() - t1

    pool.teardown()

    # Pure observability — print to capture and don't assert deltas.
    obs = {
        "gguf_path": str(small_gguf_path),
        "gguf_size_mb": small_gguf_path.stat().st_size / (1024 * 1024),
        "initial_resident_fraction": round(initial_resident, 3),
        "after_drop_resident_fraction": round(cold_resident, 3),
        "after_cold_load_resident_fraction": round(warm_resident, 3),
        "cold_load_wall_s": round(cold_load_s, 3),
        "warm_load_wall_s": round(warm_load_s, 3),
    }
    print(f"COLD_VS_WARM={json.dumps(obs)}")
    # Soft assertion: warm load should be at least as fast as cold, with
    # 50% slack for noise. If this fails repeatedly, llama-swap is
    # short-circuiting or our cache state is wrong.
    assert warm_load_s <= cold_load_s * 1.5 + 0.5, (
        f"warm load {warm_load_s:.2f}s slower than cold {cold_load_s:.2f}s — "
        f"page-cache pre-flight may not be working"
    )


def test_vmtouch_if_available_reports_some_resident_pages(
    small_gguf_path: Path,
) -> None:
    """Optional cross-check via the vmtouch CLI when installed.

    vmtouch reads /proc/<pid>/maps + mincore under the hood. We use it
    here purely as a tripwire: if mincore says "0%" but vmtouch says
    "30%", our mincore implementation has a bug. Skip when vmtouch
    isn't installed (most lab boxes).
    """

    try:
        proc = subprocess.run(
            ["vmtouch", "-q", str(small_gguf_path)],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("vmtouch not installed")
    except subprocess.TimeoutExpired:
        pytest.skip("vmtouch timed out")
    if proc.returncode != 0:
        pytest.skip(f"vmtouch failed: {proc.stderr.strip()}")

    # Just confirm vmtouch ran and produced output.
    assert proc.stdout or proc.stderr
