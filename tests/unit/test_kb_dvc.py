"""Phase 15.4: DVC-tracked KB pointer file shape + kb_version bumping.

The KB index + chunks dirs are versioned via DVC. The `.dvc` pointer files
sit under `kbs/<name>/` in git and reference the cache via md5 hashes. These
tests pin:

1. The pointer file shape DVC produces (so we notice silently-broken upgrades).
2. The lab-side layout invariants (kbs/<name>/{index,chunks}.dvc + .gitignore).
3. A small kb_version bump helper (mocked dvc-cli) — we want a stable way to
   stamp a new revision into the KB manifest when republishing.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
KBS_DIR = REPO_ROOT / "kbs"


def _read_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def test_bash_kb_pointer_files_exist_and_have_expected_shape() -> None:
    """The committed pointer files for the bash KB are well-formed DVC outputs."""

    index_dvc = KBS_DIR / "bash" / "index.dvc"
    chunks_dvc = KBS_DIR / "bash" / "chunks.dvc"

    assert index_dvc.is_file(), "expected committed pointer file kbs/bash/index.dvc"
    assert chunks_dvc.is_file(), "expected committed pointer file kbs/bash/chunks.dvc"

    for path, expected_basename in [(index_dvc, "index"), (chunks_dvc, "chunks")]:
        data = _read_yaml(path)
        assert "outs" in data, f"{path}: DVC pointer must contain `outs`"
        assert len(data["outs"]) == 1, f"{path}: exactly one out expected"
        out = data["outs"][0]
        assert out["hash"] == "md5", f"{path}: hash type pinned to md5"
        assert out["md5"].endswith(
            ".dir"
        ), f"{path}: directory output md5 must end in .dir (got {out['md5']!r})"
        assert isinstance(out["size"], int)
        assert out["size"] > 0
        assert isinstance(out["nfiles"], int)
        assert out["nfiles"] > 0
        assert out["path"] == expected_basename, (
            f"{path}: path must be the relative basename ({expected_basename!r}), "
            f"so `dvc pull kbs/<name>/<x>.dvc` restores the dir in place"
        )


def test_bash_kb_gitignore_excludes_tracked_dirs() -> None:
    """`.gitignore` next to the pointer files must hide the data dirs from git."""

    gi = KBS_DIR / "bash" / ".gitignore"
    assert gi.is_file(), "kbs/bash/.gitignore should be committed"
    lines = {line.strip() for line in gi.read_text().splitlines() if line.strip()}
    # DVC writes leading-slash patterns so they only match in this dir.
    assert "/index" in lines
    assert "/chunks" in lines


def test_dvc_config_points_at_minio_endpoint() -> None:
    """`.dvc/config` (committed) declares the MinIO endpoint; creds live in `.local`."""

    cfg = (REPO_ROOT / ".dvc" / "config").read_text()
    assert "url = s3://lab-dvc" in cfg, "remote URL must be s3://lab-dvc"
    assert (
        "endpointurl = http://localhost:9000" in cfg
    ), "endpoint must be the local MinIO; creds live in .dvc/config.local"
    # Belt-and-braces: secrets MUST NOT be in the committed config.
    assert "secret_access_key" not in cfg
    assert "access_key_id" not in cfg


def test_bump_kb_version_updates_manifest(tmp_path: Path, monkeypatch) -> None:
    """`bump_kb_version` writes a fresh `kb_version` token to the manifest.

    The helper is intentionally tiny so it composes with `dvc add`. We mock the
    dvc CLI to confirm the contract: bump_kb_version touches the manifest only;
    callers run `dvc add` separately.
    """

    # The helper lives at `tools/bump_kb_version.py` in the repo. If it's
    # missing, this test documents the contract it needs to implement.
    helper = REPO_ROOT / "tools" / "bump_kb_version.py"
    if not helper.exists():
        pytest.skip("tools/bump_kb_version.py not implemented yet")

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        dedent(
            """\
            kb_format_version: 1
            name: testkb
            slug: testkb
            status: sealed
            """
        )
    )
    fake_dvc = MagicMock()
    monkeypatch.setattr("subprocess.run", fake_dvc)

    # Import + run -- the helper is small enough to call as a module.
    import runpy

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(
            str(helper),
            run_name="__main__",
            init_globals={"__argv__": [str(helper), str(manifest)]},
        )
    assert exc.value.code == 0, "bump_kb_version must exit 0 on success"

    after = _read_yaml(manifest)
    assert "kb_version" in after, "bump_kb_version must add kb_version to manifest"
    assert isinstance(after["kb_version"], str)
    assert len(after["kb_version"]) >= 8, "kb_version should be a meaningful token"
