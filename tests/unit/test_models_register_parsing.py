"""Tests for `_parse_local` + `_local_litellm_id` — Ollama tag heuristics."""

from __future__ import annotations

from lab.models.register import _local_litellm_id, _parse_local, _split_variant_quant


def _entry(tag: str, size: int = 1_000_000_000) -> dict[str, object]:
    return {"name": tag, "digest": "sha256:deadbeef", "details": {}, "size": size}


def test_split_variant_quant_basic() -> None:
    assert _split_variant_quant("14b-q4_K_M") == ("14b", "q4_K_M", False)
    assert _split_variant_quant("8b") == ("8b", "", False)
    assert _split_variant_quant("latest") == ("latest", "", False)


def test_split_variant_quant_drops_modality_tokens() -> None:
    # `instruct` / `it` / `chat` are modality markers, not quant
    assert _split_variant_quant("8b-instruct-q4_K_M") == ("8b", "q4_K_M", False)
    assert _split_variant_quant("12b-it-q4_K_M") == ("12b", "q4_K_M", False)


def test_split_variant_quant_cloud_suffix() -> None:
    # Trailing `-cloud` flips is_cloud and is stripped from the parse
    assert _split_variant_quant("20b-cloud") == ("20b", "", True)
    assert _split_variant_quant("120b-cloud") == ("120b", "", True)
    assert _split_variant_quant("235b-cloud") == ("235b", "", True)


def test_local_litellm_id_no_trailing_dash() -> None:
    # Regression: previously produced "qwen3-vl-235b-cloud-" or similar
    assert _local_litellm_id("qwen3-vl", "235b", "", is_cloud=True) == "qwen3-vl-235b-cloud"
    assert _local_litellm_id("phi4", "latest", "") == "phi4"  # friendly map hit
    assert _local_litellm_id("qwen3", "14b", "q4_K_M") == "qwen3-14b-q4"  # friendly hit
    assert not _local_litellm_id("anything", "8b", "q4_K_M").endswith("-")


def test_parse_local_cloud_backend_tag() -> None:
    """Cloud-suffixed tags pulled via `ollama list` should be tagged ollama-cloud,
    NOT ollama-local (the old heuristic mis-classified them)."""
    parsed = _parse_local(_entry("qwen3-vl:235b-cloud"))
    assert parsed is not None
    assert parsed["backend"] == "ollama-cloud"
    assert parsed["litellm_id"] == "qwen3-vl-235b-cloud"
    assert not parsed["litellm_id"].endswith("-")


def test_parse_local_latest_tag() -> None:
    parsed = _parse_local(_entry("phi4:latest"))
    assert parsed is not None
    assert parsed["backend"] == "ollama-local"
    assert parsed["litellm_id"] == "phi4"
    assert parsed["variant"] == "latest"


def test_parse_local_q4_K_M_tag() -> None:
    parsed = _parse_local(_entry("qwen3:14b-q4_K_M"))
    assert parsed is not None
    assert parsed["backend"] == "ollama-local"
    assert parsed["litellm_id"] == "qwen3-14b-q4"
    assert parsed["quant"] == "Q4_K_M"


def test_parse_local_instruct_q4_tag() -> None:
    parsed = _parse_local(_entry("llama3.1:8b-instruct-q4_K_M"))
    assert parsed is not None
    assert parsed["litellm_id"] == "llama3.1-8b-q4"


def test_parse_local_publisher_split() -> None:
    parsed = _parse_local(_entry("library/qwen3:8b"))
    assert parsed is not None
    assert parsed["publisher"] == "library"
    assert parsed["name"] == "qwen3"


def test_parse_local_bare_tag_has_no_publisher_prefix() -> None:
    parsed = _parse_local(_entry("qwen3:8b"))
    assert parsed is not None
    assert parsed["publisher"] == "ollama"
    assert parsed["name"] == "qwen3"


def test_parse_local_empty_tag_skipped() -> None:
    assert _parse_local({"name": ""}) is None
