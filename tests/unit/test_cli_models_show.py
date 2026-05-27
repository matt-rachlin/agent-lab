"""Tests for `lab models show <litellm_id>` CLI command (Phase 19a).

Mocks psycopg.connect so no live DB is required. Verifies:
- happy path renders a table with the model's fields
- missing model exits with code 2 and a clear message
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from lab.cli import app


def _fake_row() -> tuple:
    """One row matching the column order in models_show()."""
    return (
        114,
        "qwen",
        "qwen3-30b-a3b",
        "30b",
        "Q4_K_M",
        "llama.cpp",
        "qwen3-30b-a3b-moe",
        "https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF",
        None,
        17,
        40960,
        None,
        "apache-2.0",
        ["moe", "reasoning", "tool_call"],
        "Phase 19a headline local model.",
        datetime(2026, 5, 27, tzinfo=UTC),
        None,
        "runs:/c47f48f1bcee4d3793adc601d6e0da1c",
    )


def _fake_description() -> list[tuple]:
    return [
        ("model_id",),
        ("publisher",),
        ("name",),
        ("variant",),
        ("quant",),
        ("backend",),
        ("litellm_id",),
        ("source_url",),
        ("ollama_tag",),
        ("vram_gb",),
        ("context_max",),
        ("output_max",),
        ("license",),
        ("capabilities",),
        ("notes",),
        ("pulled_at",),
        ("retired_at",),
        ("mlflow_model_uri",),
    ]


def _patched_connect(row: tuple | None):
    """Return a context-manager mock chain that mimics psycopg's API."""
    cur = MagicMock()
    cur.fetchone.return_value = row
    cur.description = _fake_description() if row else None
    cur.__enter__.return_value = cur
    cur.__exit__.return_value = False

    conn = MagicMock()
    conn.cursor.return_value = cur
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    return MagicMock(return_value=conn)


def test_models_show_happy_path() -> None:
    runner = CliRunner()
    with patch("psycopg.connect", _patched_connect(_fake_row())):
        result = runner.invoke(app, ["models", "show", "qwen3-30b-a3b-moe"])
    assert result.exit_code == 0, result.stdout
    out = result.stdout
    # Spot-check the rendered table for canonical fields the user expects.
    assert "qwen3-30b-a3b-moe" in out
    assert "llama.cpp" in out
    assert "apache-2.0" in out
    assert "moe" in out
    assert "runs:/c47f48f1bcee4d3793adc601d6e0da1c" in out


def test_models_show_missing_exits_with_code_2() -> None:
    runner = CliRunner()
    with patch("psycopg.connect", _patched_connect(None)):
        result = runner.invoke(app, ["models", "show", "no-such-model"])
    assert result.exit_code == 2
    assert "no models row" in result.stdout
    assert "no-such-model" in result.stdout
