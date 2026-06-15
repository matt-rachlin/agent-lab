"""Integration test: llama-swap is up and advertises the configured models.

Skips cleanly when:
  * the service isn't reachable on 127.0.0.1:8080 (this test isn't allowed
    to start it — that's the systemd unit's job).

When reachable, asserts:
  * ``GET /v1/models`` returns HTTP 200 with an OpenAI-shaped payload.
  * Every model listed in ``conf/serving/llama-swap.yaml`` is present in the
    response (catches drift between the config and the live service).
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
import yaml

DEFAULT_URL = "http://127.0.0.1:8080"
URL_ENV_VAR = "LAB_LLAMA_SWAP_URL"


def _service_url() -> str:
    return os.environ.get(URL_ENV_VAR, DEFAULT_URL)


def _service_up(url: str) -> bool:
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{url}/v1/models")
        return resp.status_code == 200
    except Exception:
        return False


def _expected_model_ids() -> set[str]:
    """Parse conf/serving/llama-swap.yaml and return the set of declared model IDs."""
    config_path = Path(__file__).resolve().parents[2] / "conf" / "llama-swap.yaml"
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    models = cfg.get("models", {}) or {}
    # Drop unlisted entries — they intentionally don't show up in /v1/models.
    return {
        mid
        for mid, meta in models.items()
        if isinstance(meta, dict) and not meta.get("unlisted", False)
    }


@pytest.mark.integration
def test_llama_swap_health_or_skip() -> None:
    url = _service_url()
    if not _service_up(url):
        pytest.skip(f"llama-swap not reachable at {url}/v1/models")

    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{url}/v1/models")
    assert resp.status_code == 200, f"unexpected status {resp.status_code}: {resp.text}"

    payload = resp.json()
    assert payload.get("object") == "list", f"unexpected payload shape: {payload}"
    data = payload.get("data") or []
    advertised = {row["id"] for row in data if isinstance(row, dict) and "id" in row}

    expected = _expected_model_ids()
    missing = expected - advertised
    assert not missing, (
        f"llama-swap is missing models declared in conf/serving/llama-swap.yaml: {sorted(missing)}; "
        f"advertised={sorted(advertised)}"
    )
