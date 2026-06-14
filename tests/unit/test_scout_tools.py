"""Scout tools (ADR-011): SSRF guard + validation + arxiv/web parse (no network)."""

from lab.scout_tools import _is_public_host, arxiv_search, scout_add_tool, web_search


def test_ssrf_blocks_private_and_loopback():
    assert _is_public_host("localhost") is False
    assert _is_public_host("127.0.0.1") is False
    assert _is_public_host("169.254.169.254") is False  # link-local metadata
    assert _is_public_host("10.0.0.1") is False


def test_scout_add_rejects_bad_category_and_confidence():
    assert scout_add_tool("https://x", "t", "bogus", "w")["result"] == "error"
    assert scout_add_tool("https://x", "t", "paper", "w", confidence="0.9")["result"] == "error"


def test_arxiv_parse_handles_atom(monkeypatch):
    sample = (
        "<feed><entry><title>A Paper</title>"
        "<id>http://arxiv.org/abs/2601.00001</id>"
        "<summary>We do  things.</summary></entry></feed>"
    )

    class _R:
        text = sample

    monkeypatch.setattr("lab.scout_tools._safe_get", lambda *a, **k: _R())
    out = arxiv_search("x", 5)
    assert out[0]["title"] == "A Paper"
    assert out[0]["url"] == "http://arxiv.org/abs/2601.00001"
    assert out[0]["summary"] == "We do things."


def test_web_search_parses_searxng_json(monkeypatch):
    """web_search hits the trusted local SearXNG directly (no SSRF guard) and
    maps results to title/url/content; bad categories fall back to general."""
    captured = {}

    class _R:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "results": [
                    {"title": "T1", "url": "https://a.example/1", "content": "c1"},
                    {"title": "T2", "url": "https://b.example/2", "content": None},
                ]
            }

    def _fake_get(url, params=None, **kwargs):
        captured["url"] = url
        captured["params"] = params
        return _R()

    monkeypatch.setattr("lab.scout_tools.httpx.get", _fake_get)
    out = web_search("agentic tool calling", max_results=5, categories="bogus")
    assert captured["params"]["format"] == "json"
    assert captured["params"]["categories"] == "general"  # bogus -> general
    assert [r["title"] for r in out] == ["T1", "T2"]
    assert out[0]["url"] == "https://a.example/1"
    assert out[1]["content"] == ""  # None coerced to empty


def test_web_search_returns_error_on_failure(monkeypatch):
    def _boom(*a, **k):
        raise __import__("httpx").ConnectError("down")

    monkeypatch.setattr("lab.scout_tools.httpx.get", _boom)
    out = web_search("x")
    assert "error" in out[0]
