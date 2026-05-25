"""Tests for SweepConfig.model_defaults parsing."""

from __future__ import annotations

from pathlib import Path

from lab.sweep.config import ModelDefaults, SweepConfig, load_sweep

SWEEP_YAML = """
experiment:
  slug: TEST-MD
  title: t
tasks:
  suite: smoke
models: [qwen3-14b-q4, phi4]
configs:
  - {name: greedy, temperature: 0.0, top_p: 1.0}
seeds: [1]
model_defaults:
  qwen3-14b-q4:
    system_prompt: "/no_think"
"""


def test_model_defaults_parses(tmp_path: Path) -> None:
    p = tmp_path / "sweep.yaml"
    p.write_text(SWEEP_YAML, encoding="utf-8")
    spec = load_sweep(p)
    assert "qwen3-14b-q4" in spec.model_defaults
    md = spec.model_defaults["qwen3-14b-q4"]
    assert isinstance(md, ModelDefaults)
    assert md.system_prompt == "/no_think"
    # Other model gets no override
    assert "phi4" not in spec.model_defaults


def test_model_defaults_optional() -> None:
    spec = SweepConfig.model_validate(
        {
            "experiment": {"slug": "X"},
            "tasks": {"suite": "smoke"},
            "models": ["m1"],
            "configs": [{"name": "g"}],
            "seeds": [1],
        }
    )
    assert spec.model_defaults == {}
