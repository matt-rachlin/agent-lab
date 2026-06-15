"""Canonical-prompt registry.

Task YAMLs reference prompts by ID (``system_prompt_id: agent_system_v1``)
instead of inlining the prompt body. Reading the prompt is delegated to
:class:`PromptRegistry`, which scans ``prompts/library/<prompt_id>.md``
files and exposes a small lookup API.

The on-disk format is doc-meta-compliant Markdown:

    ---
    doc_id: prompt-agent-system-v1
    title: Agent system prompt v1
    zone: lab
    kind: prompt
    status: active
    owner: m
    created: 2026-05-27
    last_updated: 2026-05-27
    tags: [lab, prompt, agent, system]
    ---

    You are a careful research assistant...

The registry derives the canonical ``prompt_id`` and integer ``version``
from the ``doc_id``. The doc-meta convention is
``prompt-<id-kebab>-v<N>``; the parser collapses the kebab id to
snake-case and keeps the version suffix as part of the prompt_id (so
``agent_system_v1`` and ``agent_system_v2`` are distinct ids), with the
integer ``N`` available separately as :attr:`PromptMeta.version`.

Example::

    doc_id: prompt-agent-system-v1
      -> prompt_id="agent_system_v1", version=1

The ``version: N`` field in frontmatter is checked against the suffix as
a consistency guard — they must agree, otherwise the loader raises.

A ``version`` keyword override on :meth:`PromptRegistry.get` picks a
specific revision when multiple files share a base id (i.e.
``agent_system_v1`` and ``agent_system_v2`` resolve to a base lookup
``agent_system``); without ``version`` the highest is returned. To pin
to an exact file, use the full ``prompt_id`` including the ``_vN``
suffix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "PromptMeta",
    "PromptNotFoundError",
    "PromptRegistry",
    "default_registry_root",
]


DEFAULT_PROMPTS_ROOT = Path("prompts/library")

# Frontmatter regex matches a leading "---\n...\n---" block. Generous
# whitespace handling so editor variations don't break parsing.
_FRONTMATTER_RE = re.compile(
    r"\A---\r?\n(?P<body>.*?)\r?\n---\r?\n?",
    re.DOTALL,
)

# A doc_id like "prompt-agent-system-v1" -> prompt_id="agent_system_v1",
# version=1. The "prompt-" prefix is required; the "-vN" suffix is
# required; the middle is the canonical prompt id with hyphens converted
# to underscores. This matches the rule documented in prompts/README.md.
_DOC_ID_RE = re.compile(r"^prompt-(?P<id>[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)-v(?P<version>\d+)$")


def default_registry_root() -> Path:
    """The canonical on-disk location of the prompts library."""
    return DEFAULT_PROMPTS_ROOT


@dataclass(frozen=True)
class PromptMeta:
    """Metadata for one prompt file, plus the rendered body."""

    # Canonical version-suffixed id, e.g. "agent_system_v1".
    prompt_id: str
    # Base (versionless) id used for "latest" lookups, e.g. "agent_system".
    base_id: str
    version: int
    title: str
    path: Path
    body: str
    tags: list[str] = field(default_factory=list)
    raw_meta: dict[str, Any] = field(default_factory=dict)


class PromptNotFoundError(KeyError):
    """Raised when :meth:`PromptRegistry.get` cannot find a prompt."""


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return ``(frontmatter_dict, body)`` for a doc-meta markdown file.

    Raises :class:`ValueError` if the file has no frontmatter or the YAML
    block is malformed.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError("missing YAML frontmatter")
    raw = yaml.safe_load(m.group("body")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"frontmatter is not a mapping: {type(raw).__name__}")
    body = text[m.end() :].lstrip("\n")
    return raw, body


def _derive_ids_and_version(doc_id: str) -> tuple[str, str, int]:
    """Convert a doc_id like 'prompt-agent-system-v1' to (id, base_id, ver).

    ``id`` is the canonical, version-suffixed prompt_id (e.g.
    ``agent_system_v1``). ``base_id`` is the same without the ``_v<N>``
    suffix (``agent_system``), used for "latest version" lookups.
    """
    m = _DOC_ID_RE.match(doc_id)
    if not m:
        raise ValueError(f"doc_id {doc_id!r} does not match 'prompt-<id>-v<N>'")
    base = m.group("id").replace("-", "_")
    version = int(m.group("version"))
    full = f"{base}_v{version}"
    return full, base, version


def _load_one(path: Path) -> PromptMeta:
    """Parse one prompt file into a :class:`PromptMeta`."""
    text = path.read_text(encoding="utf-8")
    meta, body = _split_frontmatter(text)
    if meta.get("kind") != "prompt":
        raise ValueError(f"{path}: expected frontmatter kind 'prompt', got {meta.get('kind')!r}")
    doc_id = str(meta.get("doc_id") or "")
    prompt_id, base_id, version = _derive_ids_and_version(doc_id)
    # Consistency: an explicit frontmatter `version: N` must agree with
    # the -v<N> suffix on the doc_id (one source of truth, two surfaces).
    fm_version = meta.get("version")
    if fm_version is not None and int(fm_version) != version:
        raise ValueError(
            f"{path}: frontmatter version={fm_version} disagrees with doc_id suffix v{version}"
        )
    title = str(meta.get("title") or prompt_id)
    tags_raw = meta.get("tags") or []
    if not isinstance(tags_raw, list):
        raise ValueError(f"{path}: tags must be a list")
    tags = [str(t) for t in tags_raw]
    return PromptMeta(
        prompt_id=prompt_id,
        base_id=base_id,
        version=version,
        title=title,
        path=path,
        body=body,
        tags=tags,
        raw_meta=meta,
    )


class PromptRegistry:
    """Read-only registry over ``prompts/library/*.md``.

    Supports two lookup styles:

    * Versioned: ``registry.get("agent_system_v1")`` returns that exact
      revision.
    * Versionless (base): ``registry.get("agent_system")`` returns the
      highest-version prompt sharing that base id.

    Pass ``version=N`` to either form to pin to a specific revision.
    """

    def __init__(self, root: Path | None = None) -> None:
        self._root = root if root is not None else DEFAULT_PROMPTS_ROOT
        self._by_base: dict[str, dict[int, PromptMeta]] | None = None

    @property
    def root(self) -> Path:
        return self._root

    def _ensure_loaded(self) -> dict[str, dict[int, PromptMeta]]:
        if self._by_base is not None:
            return self._by_base
        index: dict[str, dict[int, PromptMeta]] = {}
        if not self._root.exists():
            self._by_base = index
            return index
        for path in sorted(self._root.glob("*.md")):
            meta = _load_one(path)
            index.setdefault(meta.base_id, {})[meta.version] = meta
        self._by_base = index
        return index

    def reload(self) -> None:
        """Drop the cache; next call will re-scan disk."""
        self._by_base = None

    def list(self) -> list[PromptMeta]:
        """Return every loaded prompt, sorted by (base_id, version)."""
        idx = self._ensure_loaded()
        out: list[PromptMeta] = []
        for base_id in sorted(idx):
            for v in sorted(idx[base_id]):
                out.append(idx[base_id][v])
        return out

    def get(self, prompt_id: str, *, version: int | None = None) -> str:
        """Return the prompt body for ``prompt_id``.

        ``prompt_id`` may be either a base id (``agent_system``) or a
        version-suffixed id (``agent_system_v1``). Without ``version``,
        returns the latest revision. With ``version``, pins to that
        revision (or raises :class:`PromptNotFoundError`).
        """
        return self.get_meta(prompt_id, version=version).body

    def _resolve(self, prompt_id: str) -> tuple[str, int | None]:
        """Split ``prompt_id`` into ``(base_id, explicit_version)``.

        If ``prompt_id`` ends in ``_v<N>`` the suffix is treated as an
        explicit version pin and stripped; otherwise it is the base id.
        """
        m = re.fullmatch(r"(?P<base>.+)_v(?P<v>\d+)", prompt_id)
        if m:
            return m.group("base"), int(m.group("v"))
        return prompt_id, None

    def get_meta(self, prompt_id: str, *, version: int | None = None) -> PromptMeta:
        """Return the :class:`PromptMeta` for ``prompt_id``."""
        idx = self._ensure_loaded()
        base_id, suffix_version = self._resolve(prompt_id)
        # Explicit `version=` overrides a suffix pin.
        target_version = version if version is not None else suffix_version
        versions = idx.get(base_id)
        if not versions:
            raise PromptNotFoundError(
                f"no prompt registered with id {prompt_id!r} (searched under {self._root})"
            )
        if target_version is None:
            chosen = max(versions)
        else:
            if target_version not in versions:
                raise PromptNotFoundError(
                    f"prompt {prompt_id!r} has no version {target_version}; "
                    f"available: {sorted(versions)}"
                )
            chosen = target_version
        return versions[chosen]

    def has(self, prompt_id: str) -> bool:
        """Test whether ``prompt_id`` is registered (any version)."""
        idx = self._ensure_loaded()
        base_id, suffix_version = self._resolve(prompt_id)
        versions = idx.get(base_id)
        if not versions:
            return False
        if suffix_version is None:
            return True
        return suffix_version in versions
