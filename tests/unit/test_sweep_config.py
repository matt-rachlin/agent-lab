"""SweepConfig parsing + hashing tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lab.sweep.config import (
    RunConfig,
    SweepConfig,
    config_hash,
    load_sweep,
    run_id,
)

SWEEP_YAML = """
experiment:
  slug: TEST-001
  title: t
tasks:
  suite: smoke
models: [m1, m2]
configs:
  - {name: greedy, temperature: 0.0, top_p: 1.0}
  - {name: sampled, temperature: 0.7, top_p: 0.9}
seeds: [1, 2, 3]
"""


def test_load_sweep_from_yaml(tmp_path: Path) -> None:
    p = tmp_path / "sweep.yaml"
    p.write_text(SWEEP_YAML, encoding="utf-8")
    spec = load_sweep(p)
    assert spec.experiment.slug == "TEST-001"
    assert spec.models == ["m1", "m2"]
    assert len(spec.configs) == 2
    assert spec.seeds == [1, 2, 3]


def test_extra_forbids_unknown_fields() -> None:
    raw = yaml.safe_load(SWEEP_YAML)
    raw["surprise"] = "rejected"
    with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
        SweepConfig.model_validate(raw)


def test_config_hash_stable() -> None:
    cfg_a = RunConfig(name="x", temperature=0.0, top_p=1.0)
    cfg_b = RunConfig(name="y", temperature=0.0, top_p=1.0)  # different name, same params
    cfg_c = RunConfig(name="x", temperature=0.7, top_p=1.0)
    assert config_hash(cfg_a) == config_hash(cfg_b), "name should not affect hash"
    assert config_hash(cfg_a) != config_hash(cfg_c), "param change must change hash"
    # Length: 16-hex prefix
    assert len(config_hash(cfg_a)) == 16


def test_run_id_deterministic() -> None:
    a = run_id("EXP", "m1", "task-1", "abc123", 7)
    b = run_id("EXP", "m1", "task-1", "abc123", 7)
    c = run_id("EXP", "m1", "task-1", "abc123", 8)  # different seed
    d = run_id("EXP", "m2", "task-1", "abc123", 7)  # different model
    assert a == b
    assert a != c
    assert a != d
    assert len(a) == 24


def test_run_id_prompt_affects_hash() -> None:
    base = run_id("EXP", "m1", "t1", "h", 1)
    with_p = run_id("EXP", "m1", "t1", "h", 1, prompt_family="verbose", prompt_version="v1")
    assert base != with_p
