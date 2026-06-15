"""Smoke test: import the Ask The Lab page without SyntaxError / ImportError.

Streamlit pages can't be tested functionally without a running server; an
import-level smoke confirms the file is syntactically valid and its imports
resolve (modulo optional lab packages which we stub).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

PAGE_PATH = APP_DIR / "pages" / "06_Ask_The_Lab.py"


def test_page_file_exists() -> None:
    assert PAGE_PATH.exists(), f"page not found at {PAGE_PATH}"


def test_page_imports_without_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import the page module under a stubbed streamlit; verify no SyntaxError."""
    # Stub out streamlit so no server is needed.
    st_stub = MagicMock()
    st_stub.set_page_config = MagicMock()
    st_stub.title = MagicMock()
    st_stub.caption = MagicMock()
    st_stub.text_area = MagicMock(return_value="")
    st_stub.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
    st_stub.text_input = MagicMock(return_value="qwen3-4b-ft-toolcall-q4-latest")
    st_stub.number_input = MagicMock(return_value=16)
    st_stub.button = MagicMock(return_value=False)

    monkeypatch.setitem(sys.modules, "streamlit", st_stub)

    # Also stub lab.synthesizer so the import doesn't fail when lab packages
    # aren't installed in the eval-dashboard venv.
    synth_stub = types.ModuleType("lab.synthesizer")
    synth_stub.synthesize = MagicMock(
        return_value={"answer": "ok", "citations": [], "tool_calls": 0, "stop": "done"}
    )  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lab.synthesizer", synth_stub)

    import importlib.util

    spec = importlib.util.spec_from_file_location("ask_the_lab_page", PAGE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Should not raise.
    spec.loader.exec_module(module)  # type: ignore[union-attr]
