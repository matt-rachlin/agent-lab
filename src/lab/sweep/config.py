"""SweepConfig — declarative specification of a comparison sweep."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ExperimentRef(BaseModel):
    """Reference to (or definition of) an experiment row to attach runs to."""

    model_config = ConfigDict(extra="forbid")

    slug: str
    title: str | None = None
    hypothesis: str | None = None
    plan_path: str | None = None
    create_if_missing: bool = True


class TaskRef(BaseModel):
    """Which tasks to run against."""

    model_config = ConfigDict(extra="forbid")

    suite: str
    slugs: list[str] | None = None  # None = all in suite


class PromptRef(BaseModel):
    """One named prompt family to include in the sweep matrix."""

    model_config = ConfigDict(extra="forbid")

    family: str
    version: str
    content: str | None = None  # If None, looked up by (family, version) from DB


class RunConfig(BaseModel):
    """A single config-cell in the sweep matrix."""

    model_config = ConfigDict(extra="allow")

    name: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int | None = None
    scaffold: Literal["single_turn", "react", "plan_execute"] = "single_turn"
    extra: dict[str, Any] = Field(default_factory=dict)


class SweepConfig(BaseModel):
    """Top-level sweep specification (loaded from YAML)."""

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentRef
    tasks: TaskRef
    models: list[str]  # litellm_id values
    prompts: list[PromptRef] | None = None
    configs: list[RunConfig]
    seeds: list[int]

    judges: list[str] | None = None  # litellm_ids to use as judges in Phase 2+ (ignored Phase 1)
    max_concurrency: int = 1  # local sweeps: 1 (single GPU); cloud: up to 3 (Pro tier)
    request_timeout_sec: int = 600


def load_sweep(path: Path) -> SweepConfig:
    """Load a sweep config from YAML or JSON."""
    text = path.read_text(encoding="utf-8")
    raw: Any = yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
    return SweepConfig.model_validate(raw)


def config_hash(config: RunConfig) -> str:
    """Stable hash of the sampleable parameters of a RunConfig.

    Excludes `name` (cosmetic) — two RunConfigs with the same parameters and
    different names should hash to the same value.
    """
    payload = {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "scaffold": config.scaffold,
        "extra": config.extra,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def run_id(
    experiment_slug: str,
    model_litellm_id: str,
    task_slug: str,
    config_hash_str: str,
    seed: int,
    prompt_family: str | None = None,
    prompt_version: str | None = None,
) -> str:
    """Deterministic hash for a (cell) run.

    Identical inputs → identical run_id, so retries are idempotent and a
    resumable sweep can simply skip rows whose run_id already exists.
    """
    parts = [
        experiment_slug,
        model_litellm_id,
        task_slug,
        config_hash_str,
        str(seed),
        prompt_family or "default",
        prompt_version or "default",
    ]
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]
