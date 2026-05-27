"""Tests for `check_litellm_keep_alive` preflight."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from lab.sweep.preflight import PreflightError, check_litellm_keep_alive


def _write(path: Path, cfg: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_passes_when_all_locals_have_keep_alive(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "c.yaml",
        {
            "model_list": [
                {
                    "model_name": "qwen3-14b-q4",
                    "litellm_params": {
                        "model": "ollama_chat/qwen3:14b-q4_K_M",
                        "api_base": "http://x:11434",
                        "keep_alive": "5m",
                    },
                }
            ]
        },
    )
    check_litellm_keep_alive(p)  # no raise


def test_raises_when_local_missing_keep_alive(tmp_path: Path) -> None:
    p = _write(
        tmp_path / "c.yaml",
        {
            "model_list": [
                {
                    "model_name": "qwen3-14b-q4",
                    "litellm_params": {
                        "model": "ollama_chat/qwen3:14b-q4_K_M",
                        "api_base": "http://x:11434",
                        # no keep_alive
                    },
                },
                {
                    "model_name": "phi4",
                    "litellm_params": {
                        "model": "ollama_chat/phi4:latest",
                        "keep_alive": "5m",
                    },
                },
            ]
        },
    )
    with pytest.raises(PreflightError, match="qwen3-14b-q4"):
        check_litellm_keep_alive(p)


def test_cloud_models_not_required_to_have_keep_alive(tmp_path: Path) -> None:
    # -cloud tags are proxied by Ollama Cloud, not loaded in local VRAM
    p = _write(
        tmp_path / "c.yaml",
        {
            "model_list": [
                {
                    "model_name": "gpt-oss-120b-cloud",
                    "litellm_params": {
                        "model": "ollama_chat/gpt-oss:120b-cloud",
                        "api_base": "http://x:11434",
                    },
                }
            ]
        },
    )
    check_litellm_keep_alive(p)  # no raise


def test_missing_config_file_is_silent(tmp_path: Path) -> None:
    # If we can't discover a config, the preflight does not block.
    nonexistent = tmp_path / "nope.yaml"
    check_litellm_keep_alive(nonexistent)


def test_real_repo_config_has_keep_alive() -> None:
    """The repo's checked-in proxy config should pass preflight."""
    cfg = Path("/data/lab/code/conf/litellm-config.yaml")
    if not cfg.exists():
        pytest.skip("repo proxy config not present")
    check_litellm_keep_alive(cfg)
