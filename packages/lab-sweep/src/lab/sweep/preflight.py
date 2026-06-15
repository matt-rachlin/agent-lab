"""Sweep preflight checks.

Refuse to start a sweep when the LiteLLM proxy is configured in a way that
caused EXP-001 thrash: any local-Ollama model without an explicit `keep_alive`
will get unloaded between cells, defeating the whole sweep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class PreflightError(RuntimeError):
    """Preflight refused to start the sweep."""


# Discovery order; first existing path wins.
_DEFAULT_CONFIG_PATHS = (
    Path("/data/lab/code/conf/serving/litellm-config.yaml"),
    Path("/data/lab/code/conf/litellm/config.yaml"),
    Path("/data/lab/services/litellm-config.yaml"),
)


def _find_default_config() -> Path | None:
    for p in _DEFAULT_CONFIG_PATHS:
        if p.exists():
            return p
    return None


def _is_ollama_local(litellm_params: dict[str, Any]) -> bool:
    """A model is treated as local-Ollama if its `model` string starts with
    `ollama` and its `api_base` (when present) points at the local daemon.

    Cloud models also go through the local daemon but have `:Nb-cloud`,
    `:cloud`, or `-cloud` tags — they aren't materially affected by
    `keep_alive` because the local daemon never holds their weights in VRAM.
    Examples of cloud tags we must NOT treat as local:
        `ollama_chat/gpt-oss:20b-cloud`     (suffix `-cloud`)
        `ollama_chat/glm-5.1:cloud`         (plain `:cloud` tag)
        `ollama_chat/qwen3-coder:480b-cloud`
    """
    model = str(litellm_params.get("model", ""))
    if not model.startswith(("ollama_chat/", "ollama/")):
        return False
    tag = model.split("/", 1)[1] if "/" in model else ""
    # The piece after the colon is the actual Ollama tag.
    after_colon = tag.split(":", 1)[-1] if ":" in tag else ""
    return not (tag.endswith("-cloud") or "-cloud" in after_colon or after_colon == "cloud")


def check_litellm_keep_alive(config_path: Path | None = None) -> None:
    """Raise PreflightError if any local Ollama model lacks `keep_alive`."""
    path = config_path or _find_default_config()
    if path is None or not path.exists():
        # No discoverable config → don't block; nothing to verify.
        return
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return
    model_list = raw.get("model_list") or []
    offenders: list[str] = []
    for entry in model_list:
        if not isinstance(entry, dict):
            continue
        params = entry.get("litellm_params") or {}
        if not isinstance(params, dict):
            continue
        if not _is_ollama_local(params):
            continue
        if "keep_alive" not in params:
            offenders.append(str(entry.get("model_name") or params.get("model") or "<unknown>"))
    if offenders:
        joined = ", ".join(sorted(offenders))
        raise PreflightError(
            "LiteLLM proxy config has local Ollama models without `keep_alive`: "
            f"{joined}. Set `keep_alive: 5m` (or similar) on each "
            f"to prevent VRAM thrash between sweep cells. (config: {path})"
        )
