"""Scout tools v1 (ADR-011): SSRF guard + validation + arxiv parse (no network)."""

from lab.scout_tools import _is_public_host, arxiv_search, scout_add_tool


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
