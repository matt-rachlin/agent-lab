"""Unit tests for `lab.agent.tools.http_fetch` — schema + allow-list.

We mock the network with `httpx.MockTransport`; the actual host-list policy
is what we care about, not what example.com returns.
"""

from __future__ import annotations

import httpx
import pytest

from lab.agent.tools import http_fetch as http_fetch_mod


@pytest.fixture
def mock_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace `httpx.Client` (as seen by `http_fetch_mod`) with a mock-transport-backed shim.

    We keep a reference to the *real* `httpx.Client` to construct the inner
    client; only the attribute lookup `http_fetch_mod.httpx.Client` is
    rerouted to our shim, so the rest of the test suite is unaffected.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.com":
            return httpx.Response(200, content=b"hello\n", headers={"Content-Type": "text/plain"})
        if request.url.host == "huge.example":
            return httpx.Response(200, content=b"x" * 5000)
        return httpx.Response(404)

    real_client_cls = httpx.Client
    transport = httpx.MockTransport(handler)

    class _ClientShim:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._inner = real_client_cls(transport=transport)

        def __enter__(self) -> _ClientShim:
            return self

        def __exit__(self, *args: object) -> None:
            self._inner.close()

        def get(self, url: str) -> httpx.Response:
            return self._inner.get(url)

    monkeypatch.setattr(http_fetch_mod.httpx, "Client", _ClientShim)


def test_http_fetch_refuses_when_allowlist_empty(
    monkeypatch: pytest.MonkeyPatch, mock_httpx: None
) -> None:
    monkeypatch.delenv("LAB_HTTP_ALLOWLIST", raising=False)
    with pytest.raises(PermissionError, match="no hosts are allow-listed"):
        http_fetch_mod.http_fetch(url="https://example.com/")


def test_http_fetch_refuses_non_allowed_host(
    monkeypatch: pytest.MonkeyPatch, mock_httpx: None
) -> None:
    monkeypatch.setenv("LAB_HTTP_ALLOWLIST", "other.example")
    with pytest.raises(PermissionError, match="not in the allow-list"):
        http_fetch_mod.http_fetch(url="https://example.com/")


def test_http_fetch_allows_listed_host(monkeypatch: pytest.MonkeyPatch, mock_httpx: None) -> None:
    monkeypatch.setenv("LAB_HTTP_ALLOWLIST", "example.com,other.example")
    out = http_fetch_mod.http_fetch(url="https://example.com/")
    assert out["status"] == 200
    assert out["content"].startswith("hello")
    assert out["truncated"] is False


def test_http_fetch_rejects_non_http_scheme(
    monkeypatch: pytest.MonkeyPatch, mock_httpx: None
) -> None:
    monkeypatch.setenv("LAB_HTTP_ALLOWLIST", "example.com")
    with pytest.raises(ValueError, match="scheme"):
        http_fetch_mod.http_fetch(url="ftp://example.com/")


def test_http_fetch_caps_response_size(monkeypatch: pytest.MonkeyPatch, mock_httpx: None) -> None:
    monkeypatch.setenv("LAB_HTTP_ALLOWLIST", "huge.example")
    out = http_fetch_mod.http_fetch(url="https://huge.example/", max_bytes=100)
    assert out["truncated"] is True
    assert len(out["content"]) == 100


def test_http_fetch_rejects_url_without_host(
    monkeypatch: pytest.MonkeyPatch, mock_httpx: None
) -> None:
    monkeypatch.setenv("LAB_HTTP_ALLOWLIST", "example.com")
    with pytest.raises(ValueError, match="hostname"):
        http_fetch_mod.http_fetch(url="https:///")


def test_http_fetch_max_bytes_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_bytes must be positive"):
        http_fetch_mod.http_fetch(url="https://example.com/", max_bytes=0)


# ---------------------------------------------------------------------------
# LAB_HTTP_FIXTURE_DIR offline mode (added 6f for PBS-Agent v0.1 HTTP tasks).
# ---------------------------------------------------------------------------


def test_http_fetch_fixture_dir_serves_from_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """When LAB_HTTP_FIXTURE_DIR is set, http_fetch reads from <dir>/<host>/<path>."""

    import pathlib

    fixture_dir = pathlib.Path(tmp_path)  # type: ignore[arg-type]
    (fixture_dir / "lab.example").mkdir(parents=True)
    (fixture_dir / "lab.example" / "status.json").write_text(
        '{"service": "lab", "uptime_minutes": 4242}'
    )
    monkeypatch.setenv("LAB_HTTP_ALLOWLIST", "lab.example")
    monkeypatch.setenv("LAB_HTTP_FIXTURE_DIR", str(fixture_dir))

    out = http_fetch_mod.http_fetch(url="http://lab.example/status.json")
    assert out["status"] == 200
    assert "4242" in out["content"]
    assert out["headers"].get("x-lab-fixture") == "hit"


def test_http_fetch_fixture_dir_miss_returns_404(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    import pathlib

    fixture_dir = pathlib.Path(tmp_path)  # type: ignore[arg-type]
    (fixture_dir / "lab.example").mkdir(parents=True)
    monkeypatch.setenv("LAB_HTTP_ALLOWLIST", "lab.example")
    monkeypatch.setenv("LAB_HTTP_FIXTURE_DIR", str(fixture_dir))

    out = http_fetch_mod.http_fetch(url="http://lab.example/missing.json")
    assert out["status"] == 404
    assert out["headers"].get("x-lab-fixture") == "miss"


def test_http_fetch_fixture_dir_still_enforces_allowlist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    """Fixture mode does NOT bypass the host allow-list."""

    import pathlib

    fixture_dir = pathlib.Path(tmp_path)  # type: ignore[arg-type]
    monkeypatch.setenv("LAB_HTTP_ALLOWLIST", "lab.example")
    monkeypatch.setenv("LAB_HTTP_FIXTURE_DIR", str(fixture_dir))

    with pytest.raises(PermissionError, match="not in the allow-list"):
        http_fetch_mod.http_fetch(url="http://other.example/whatever")


def test_http_fetch_fixture_dir_refuses_path_escape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    import pathlib

    fixture_dir = pathlib.Path(tmp_path)  # type: ignore[arg-type]
    (fixture_dir / "lab.example").mkdir(parents=True)
    monkeypatch.setenv("LAB_HTTP_ALLOWLIST", "lab.example")
    monkeypatch.setenv("LAB_HTTP_FIXTURE_DIR", str(fixture_dir))

    # `..` is normalised by urlparse before we see it for many shapes, but
    # the safety check still has to refuse anything that resolves outside
    # the host dir.
    out = http_fetch_mod.http_fetch(url="http://lab.example/../etc/passwd")
    # urlparse normalises the path so this is a miss inside the host dir,
    # which is the expected safe outcome (404, not a server error).
    assert out["status"] == 404
