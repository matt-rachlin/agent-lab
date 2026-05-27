"""Unit tests for tools/new_experiment.py — no live DB (psycopg patched)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

# Load tools/new_experiment.py as a module (it's not under src/).
_SPEC = importlib.util.spec_from_file_location(
    "new_experiment",
    Path(__file__).resolve().parents[2] / "tools" / "new_experiment.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
new_experiment = importlib.util.module_from_spec(_SPEC)
sys.modules["new_experiment"] = new_experiment
_SPEC.loader.exec_module(new_experiment)


@pytest.fixture
def fake_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stage a fake repo layout with the two template files."""
    docs_exp = tmp_path / "docs" / "exp"
    conf_sweep = tmp_path / "conf" / "sweep"
    analysis = tmp_path / "analysis"
    docs_exp.mkdir(parents=True)
    conf_sweep.mkdir(parents=True)
    analysis.mkdir(parents=True)

    (docs_exp / "_template.md").write_text(
        "# EXP-NNN: title\n\nDate created: YYYY-MM-DD\n\nPlan path: docs/exp/EXP-NNN-<slug>.md\n",
        encoding="utf-8",
    )
    (conf_sweep / "_template.yaml").write_text(
        "experiment:\n  slug: EXP-NNN\n  plan_path: docs/exp/EXP-NNN-<slug>.md\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(new_experiment, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(new_experiment, "DOCS_EXP", docs_exp)
    monkeypatch.setattr(new_experiment, "CONF_SWEEP", conf_sweep)
    monkeypatch.setattr(new_experiment, "ANALYSIS", analysis)
    monkeypatch.setattr(new_experiment, "DOC_TEMPLATE", docs_exp / "_template.md")
    monkeypatch.setattr(new_experiment, "SWEEP_TEMPLATE", conf_sweep / "_template.yaml")
    return tmp_path


def _patch_psycopg_connect_returns_no_rows() -> object:
    """Return a context manager mock that yields a cursor with empty fetchall()."""
    cur = MagicMock()
    cur.fetchall.return_value = []
    cur.fetchone.return_value = None
    cur.__enter__.return_value = cur
    cur.__exit__.return_value = False
    conn = MagicMock()
    conn.cursor.return_value = cur
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    return conn


def test_validate_slug_accepts_good_slug() -> None:
    assert new_experiment._validate_slug("retrieval-ablation") == "retrieval-ablation"
    assert new_experiment._validate_slug("ABC-Foo") == "abc-foo"  # lowercased


def test_validate_slug_rejects_bad_slug() -> None:
    import typer

    for bad in ("a", "!nope", "with space", "x" * 100):
        with pytest.raises(typer.BadParameter):
            new_experiment._validate_slug(bad)


def test_next_exp_number_filesystem_only(fake_repo: Path) -> None:
    (fake_repo / "docs" / "exp" / "EXP-001-foo.md").write_text("x", encoding="utf-8")
    (fake_repo / "docs" / "exp" / "EXP-003-bar.md").write_text("x", encoding="utf-8")
    (fake_repo / "conf" / "sweep" / "EXP-002.yaml").write_text("x", encoding="utf-8")
    assert new_experiment._next_exp_number(dsn=None) == 4


def test_next_exp_number_consults_db(fake_repo: Path) -> None:
    cur = MagicMock()
    cur.fetchall.return_value = [("EXP-009",), ("EXP-010b",), ("RELIABILITY-001",)]
    cur.__enter__.return_value = cur
    cur.__exit__.return_value = False
    conn = MagicMock()
    conn.cursor.return_value = cur
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False
    with patch.object(new_experiment.psycopg, "connect", return_value=conn):
        assert new_experiment._next_exp_number(dsn="postgresql://fake") == 11


def test_dry_run_writes_nothing(fake_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(new_experiment.app, ["my-slug", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Would create" in result.output
    # No files written
    assert list((fake_repo / "docs" / "exp").glob("EXP-0*-my-slug.md")) == []
    assert list((fake_repo / "conf" / "sweep").glob("EXP-0*.yaml")) == []


def test_creates_all_artifacts(fake_repo: Path) -> None:
    runner = CliRunner()
    with patch.object(
        new_experiment.psycopg, "connect", return_value=_patch_psycopg_connect_returns_no_rows()
    ):
        result = runner.invoke(new_experiment.app, ["my-slug"])
    assert result.exit_code == 0, result.output

    docs = list((fake_repo / "docs" / "exp").glob("EXP-0*-my-slug.md"))
    sweeps = list((fake_repo / "conf" / "sweep").glob("EXP-0*.yaml"))
    assert len(docs) == 1
    assert docs[0].name == "EXP-001-my-slug.md"
    assert len(sweeps) == 1
    assert sweeps[0].name == "EXP-001.yaml"
    # gitkeep created
    assert (fake_repo / "analysis" / "EXP-001" / ".gitkeep").exists()
    # Slug substitution happened in the doc
    body = docs[0].read_text(encoding="utf-8")
    assert "EXP-001" in body
    assert "EXP-NNN" not in body
    # And in the sweep config
    sweep_body = sweeps[0].read_text(encoding="utf-8")
    assert "slug: EXP-001" in sweep_body
    assert "docs/exp/EXP-001-my-slug.md" in sweep_body


def test_refuses_to_overwrite_existing_doc(
    fake_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force _next_exp_number to return 5, then stage a collision at EXP-005-my-slug.md.
    monkeypatch.setattr(new_experiment, "_next_exp_number", lambda dsn=None: 5)
    (fake_repo / "docs" / "exp" / "EXP-005-my-slug.md").write_text("existing\n", encoding="utf-8")
    runner = CliRunner()
    with patch.object(
        new_experiment.psycopg, "connect", return_value=_patch_psycopg_connect_returns_no_rows()
    ):
        result = runner.invoke(new_experiment.app, ["my-slug"])
    assert result.exit_code == 2, result.output
    assert "already exists" in result.output
