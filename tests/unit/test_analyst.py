"""NS-1 Analyst v0 (charter NS-1, ADR-012) — read tools + LAR wiring.

No live DB and no live LLM: the read tools get a fake cursor injected with
SYNTHETIC rows (one model carries a planted F-017 artefact — mean score ~0% with
zero tool calls), and analyze_experiment runs against a MOCKED run_agent driving
a scripted tool-using trajectory. The golden assertion is that the read tools
surface the right numbers and the planted smell is detectable from the tool
outputs — proving the read tools + agent wiring, not the LLM.

The slug is now BOUND into the tools (build_tools(slug, cursor_factory=...)), so
the model never supplies it. The scripted trajectory therefore builds the tools
per-slug and invokes each impl with NO arguments, mirroring how the real runtime
dispatches a zero-parameter tool.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from lab.analyst import (
    F017_SCORE_FLOOR,
    analyst_pass_rates,
    analyst_query_experiment,
    analyst_run_metadata,
    analyze_experiment,
    build_tools,
)

# --------------------------------------------------------------------------- #
# Synthetic corpus: a good model (~75% pass) + a planted F-017 model (~0%).    #
# --------------------------------------------------------------------------- #

_GOOD = "phi-4-mini"
_SMELL = "phi-4-reasoning-plus"  # the F-017 case: reasoning model, no emission
_EVAL = "bfcl_ast_match"


class _Col:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeCursor:
    """Minimal DB-API cursor over canned result sets keyed by the leading SQL
    token group. Returns whatever the matching query was registered with."""

    def __init__(self, datasets: dict[str, tuple[list[str], list[tuple[Any, ...]]]]) -> None:
        self._datasets = datasets
        self._cols: list[str] = []
        self._rows: list[tuple[Any, ...]] = []
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    @property
    def description(self) -> list[_Col]:
        return [_Col(c) for c in self._cols]

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> None:
        self.executed.append((query, params))
        if "AVG(res.score)" in query:
            key = "pass_rates"
        elif "trace_path" in query and "tool_call_count" in query:
            key = "metadata"
        else:
            key = "runs"
        self._cols, self._rows = self._datasets[key]

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows


def _factory(
    datasets: dict[str, tuple[list[str], list[tuple[Any, ...]]]],
) -> tuple[Any, _FakeCursor]:
    cur = _FakeCursor(datasets)

    @contextlib.contextmanager
    def factory() -> Iterator[_FakeCursor]:
        yield cur

    return factory, cur


def _synthetic() -> dict[str, tuple[list[str], list[tuple[Any, ...]]]]:
    runs_cols = [
        "run_id",
        "status",
        "trust_level",
        "model",
        "backend",
        "evaluator",
        "score",
        "passed",
    ]
    runs = [
        ("r1", "done", "raw", _GOOD, "vllm", _EVAL, 1.0, True),
        ("r2", "done", "raw", _GOOD, "vllm", _EVAL, 1.0, True),
        ("r3", "done", "raw", _GOOD, "vllm", _EVAL, 0.0, False),
        ("r4", "done", "raw", _GOOD, "vllm", _EVAL, 1.0, True),
        # planted F-017 artefact: scores all 0 (no emission)
        ("r5", "done", "raw", _SMELL, "vllm", _EVAL, 0.0, False),
        ("r6", "done", "raw", _SMELL, "vllm", _EVAL, 0.0, False),
        ("r7", "done", "raw", _SMELL, "vllm", _EVAL, 0.0, False),
    ]
    pr_cols = ["model", "evaluator", "n", "mean_score", "pass_rate"]
    pass_rates = [
        (_GOOD, _EVAL, 4, 0.75, 0.75),
        (_SMELL, _EVAL, 3, 0.0, 0.0),  # <- planted smell
    ]
    meta_cols = [
        "run_id",
        "status",
        "error",
        "trace_path",
        "tool_call_count",
        "actual_turns",
        "tokens_out",
    ]
    meta = [
        ("r1", "done", None, "/t/r1", 1, 1, 40),
        ("r5", "done", None, "/t/r5", 0, 1, 330),  # F-017 tell: prose, no call
    ]
    return {
        "runs": (runs_cols, runs),
        "pass_rates": (pr_cols, pass_rates),
        "metadata": (meta_cols, meta),
    }


# --------------------------------------------------------------------------- #
# Read-tool unit tests                                                          #
# --------------------------------------------------------------------------- #


def test_query_experiment_surfaces_rows_and_decimal_coercion() -> None:
    factory, cur = _factory(_synthetic())
    out = analyst_query_experiment("EXP-NS1", cursor_factory=factory)
    assert out["found"] is True
    assert out["slug"] == "EXP-NS1"
    assert len(out["rows"]) == 7
    # slug passed as a bound parameter (no string interpolation)
    assert cur.executed[0][1] == ("EXP-NS1",)
    # scores arrive as floats, not Decimal
    assert all(isinstance(r["score"], float) for r in out["rows"])
    assert {r["model"] for r in out["rows"]} == {_GOOD, _SMELL}


def test_query_experiment_empty_when_no_rows() -> None:
    factory, _ = _factory(
        {
            "runs": (["run_id"], []),
            "pass_rates": ([], []),
            "metadata": ([], []),
        }
    )
    out = analyst_query_experiment("MISSING", cursor_factory=factory)
    assert out["found"] is False
    assert out["rows"] == []


def test_pass_rates_computes_and_flags_planted_smell() -> None:
    factory, _ = _factory(_synthetic())
    out = analyst_pass_rates("EXP-NS1", cursor_factory=factory)
    by_model = {c["model"]: c for c in out["cells"]}
    assert by_model[_GOOD]["pass_rate"] == 0.75
    assert by_model[_SMELL]["mean_score"] == 0.0
    # the planted F-017 artefact is flagged, the healthy model is not
    smell_models = {s["model"] for s in out["smells"]}
    assert smell_models == {_SMELL}
    smell = out["smells"][0]
    assert smell["smell"] == "f017"
    assert smell["evaluator"] == _EVAL
    assert smell["mean_score"] <= F017_SCORE_FLOOR


def test_pass_rates_does_not_flag_zero_n_cell() -> None:
    factory, _ = _factory(
        {
            "runs": ([], []),
            "pass_rates": (
                ["model", "evaluator", "n", "mean_score", "pass_rate"],
                [("m", "e", 0, None, None)],
            ),
            "metadata": ([], []),
        }
    )
    out = analyst_pass_rates("EXP", cursor_factory=factory)
    assert out["smells"] == []


def test_run_metadata_reads_trace_fields() -> None:
    factory, _ = _factory(_synthetic())
    out = analyst_run_metadata("EXP-NS1", cursor_factory=factory)
    runs = {r["run_id"]: r for r in out["runs"]}
    # F-017 tell available in metadata: prose tokens, zero tool calls
    assert runs["r5"]["tool_call_count"] == 0
    assert runs["r5"]["tokens_out"] == 330
    assert runs["r1"]["tool_call_count"] == 1


def test_build_tools_are_all_read_only() -> None:
    tools = build_tools("EXP-NS1")
    names = {t.name for t in tools}
    assert names == {
        "analyst_query_experiment",
        "analyst_pass_rates",
        "analyst_run_metadata",
    }
    assert all(t.side_effect == "read" for t in tools)


def test_build_tools_bind_slug_and_expose_no_slug_param() -> None:
    """The slug is bound into the impls via closures (mirroring maintainer_tools);
    the JSON parameters carry NO slug, so the model cannot supply/garble it, and
    each tool always queries the experiment it was built for."""
    factory, cur = _factory(_synthetic())
    tools = build_tools("EXP-NS1", cursor_factory=factory)
    for t in tools:
        # no slug (and indeed no required params) in the tool's JSON schema
        assert t.parameters.get("required") == []
        assert "slug" not in t.parameters.get("properties", {})
    by_name = {t.name: t for t in tools}
    # the bound impl is called with NO arguments (as the runtime would) and still
    # queries the correct, bound experiment
    out = by_name["analyst_query_experiment"].impl()
    assert out["slug"] == "EXP-NS1"
    assert cur.executed[-1][1] == ("EXP-NS1",)
    # a different slug builds tools bound to that slug instead
    factory2, cur2 = _factory(_synthetic())
    other = {t.name: t for t in build_tools("OTHER", cursor_factory=factory2)}
    out2 = other["analyst_pass_rates"].impl()
    assert out2["slug"] == "OTHER"
    assert cur2.executed[-1][1] == ("OTHER",)


# --------------------------------------------------------------------------- #
# Golden eval: analyze_experiment with run_agent MOCKED to a scripted          #
# tool-using trajectory. Proves wiring: tools surface the numbers + smell.     #
# --------------------------------------------------------------------------- #


class _FakeAgentResult:
    def __init__(self, messages: list[dict[str, Any]], tool_results: list[dict[str, Any]]) -> None:
        self.messages = messages
        self.tool_calls = len(tool_results)
        self.tool_results = tool_results
        self.stop_reason = "stop"


def test_analyze_experiment_golden_surfaces_planted_smell(monkeypatch: Any) -> None:
    """Mock run_agent to a scripted trajectory that actually invokes the real
    read tools (against synthetic data), then assert analyze_experiment lifts
    the planted F-017 smell out of the tool results and a writeup out of the
    messages. No LLM, no DB.

    The tools are now built PER-SLUG with the cursor_factory bound, and each impl
    is called with NO arguments (the slug is closed over) — exactly as the real
    runtime dispatches a zero-parameter tool. We rebuild the tools here with the
    synthetic factory because analyze_experiment's own build_tools(slug) would
    bind the real-DB cursor factory."""
    factory, _ = _factory(_synthetic())

    def fake_run_agent(**kwargs: Any) -> _FakeAgentResult:
        # confirm wiring: analyst is read-only (default allow-set, no grants)
        assert "allow_side_effects" not in kwargs
        assert kwargs["actor"] == "analyst"
        slug = "EXP-NS1"
        # Rebind the tools to the synthetic factory (the slug is bound, not passed).
        tools = {t.name: t for t in build_tools(slug, cursor_factory=factory)}
        # the model supplies NO slug — the impls are zero-arg closures
        for t in build_tools(slug, cursor_factory=factory):
            assert "slug" not in t.parameters.get("properties", {})
        results: list[dict[str, Any]] = []
        for name in ("analyst_query_experiment", "analyst_pass_rates", "analyst_run_metadata"):
            res = tools[name].impl()  # no slug arg — cannot be garbled
            results.append({"name": name, "args": {}, "result": res})
        messages = [
            {"role": "system", "content": kwargs["system"]},
            {"role": "user", "content": kwargs["user"]},
            {
                "role": "assistant",
                "content": (
                    f"{_GOOD} passes 75% on {_EVAL}. {_SMELL} scores ~0% — flagged "
                    "as an F-017 non-emission artefact; quarantine before leaderboard."
                ),
            },
        ]
        return _FakeAgentResult(messages, results)

    monkeypatch.setattr("lab.analyst.run_agent", fake_run_agent)

    out = analyze_experiment(slug="EXP-NS1")

    # golden: the planted smell is surfaced from the tool outputs
    assert out["slug"] == "EXP-NS1"
    assert out["tool_calls"] == 3
    assert out["stop"] == "stop"
    assert len(out["smells"]) == 1
    smell = out["smells"][0]
    assert smell["model"] == _SMELL
    assert smell["smell"] == "f017"
    # the healthy model is NOT flagged
    assert _GOOD not in {s["model"] for s in out["smells"]}
    # a writeup was extracted from the trajectory
    assert _SMELL in out["writeup"]
    assert "quarantine" in out["writeup"].lower()


def test_analyze_experiment_clean_experiment_has_no_smells(monkeypatch: Any) -> None:
    clean = _synthetic()
    # overwrite the smell model's aggregate with a healthy score
    clean["pass_rates"] = (
        ["model", "evaluator", "n", "mean_score", "pass_rate"],
        [(_GOOD, _EVAL, 4, 0.75, 0.75), (_SMELL, _EVAL, 3, 0.66, 0.66)],
    )
    factory, _ = _factory(clean)

    def fake_run_agent(**kwargs: Any) -> _FakeAgentResult:
        tools = {t.name: t for t in build_tools("EXP", cursor_factory=factory)}
        res = tools["analyst_pass_rates"].impl()
        results = [{"name": "analyst_pass_rates", "args": {}, "result": res}]
        messages = [{"role": "assistant", "content": "all models healthy"}]
        return _FakeAgentResult(messages, results)

    monkeypatch.setattr("lab.analyst.run_agent", fake_run_agent)
    out = analyze_experiment(slug="EXP")
    assert out["smells"] == []
    assert out["writeup"] == "all models healthy"
