"""Source plan schema. Output of discovery; input to acquisition.

Vendored from kb_builder.discovery.plan. Lab's 6h-a CLI doesn't perform
discovery — this module exists so the vendored fetchers stay importable.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

from lab.rag._util import atomic_write_text
from lab.rag.manifest import Authority, SourceType
from pydantic import BaseModel, Field
from ruamel.yaml import YAML

FETCHER_BY_TYPE: dict[SourceType, str] = {
    "manpage": "manpage",
    "git-repo": "git",
    "html-single-page": "html_single",
    "html-sitemap": "html_sitemap",
    "html-spa": "html_spa",
    "pdf": "pdf",
    "rfc": "rfc",
    "github-source-tree": "git",
    "stack-exchange": "stack_exchange",
    "webfetch": "webfetch",
}


class PlannedSource(BaseModel):
    url: str
    type: SourceType
    authority: Authority = "community"
    inclusion_rationale: str = ""
    expected_size: str | None = None  # human-friendly e.g. "~1 MB"
    paths: list[str] | None = None  # for git-repo / github-source-tree
    license: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def fetcher(self) -> str:
        return FETCHER_BY_TYPE[self.type]


class SourcePlan(BaseModel):
    kb_name: str
    topic_summary: str = ""
    subtopics: list[str] = Field(default_factory=list)
    sources: list[PlannedSource]
    notes: str | None = None


_yaml = YAML(typ="rt")
_yaml.default_flow_style = False
_yaml.width = 100
_yaml.preserve_quotes = True


def dump_plan(plan: SourcePlan) -> str:
    buf = StringIO()
    _yaml.dump(plan.model_dump(mode="json"), buf)
    return buf.getvalue()


def write_plan(path: Path, plan: SourcePlan) -> None:
    atomic_write_text(path, dump_plan(plan))


def load_plan(path: Path) -> SourcePlan:
    with path.open("r", encoding="utf-8") as fh:
        return SourcePlan.model_validate(_yaml.load(fh))
