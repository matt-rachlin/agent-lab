"""Unit tests for `lab.observability.mlflow_mirror.MlflowMirror`.

These tests mock the underlying mlflow client and do not contact the
actual MLflow server. They cover:

* the mirror is disabled when the URL env / setting is empty
* the mirror is disabled when the ping at construction time fails
* the mirror is a no-op when disabled (all public methods return None
  and never raise)
* upsert_experiment, log_run, log_finding, log_model_card all call the
  mlflow client correctly when enabled (params, metrics, tags shapes)
* idempotency: log_run with the same lab run_id is reused via the
  runName tag — only one MLflow run is created
* FAILED status maps through to MLflow's set_terminated
* exceptions raised inside the client are swallowed (the mirror logs
  to stderr but does not propagate)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from lab.observability import mlflow_mirror as mm

# ---------------------------------------------------------------------------
# Helpers — fake MlflowClient
# ---------------------------------------------------------------------------


@dataclass
class FakeRun:
    info: Any


@dataclass
class FakeInfo:
    run_id: str


@dataclass
class FakeExperiment:
    experiment_id: str
    name: str


@dataclass
class FakeClient:
    """Records every interaction; mimics MlflowClient's surface."""

    experiments: dict[str, str] = field(default_factory=dict)
    runs: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # run_id -> {tags, params, metrics, status, exp_id, run_name}
    next_id: int = 0
    raise_on: str | None = None  # name of method that should raise

    # ---- experiments ----
    def get_experiment_by_name(self, name: str) -> FakeExperiment | None:
        if name in self.experiments:
            return FakeExperiment(experiment_id=self.experiments[name], name=name)
        return None

    def create_experiment(self, name: str) -> str:
        self.next_id += 1
        eid = str(self.next_id)
        self.experiments[name] = eid
        return eid

    def set_experiment_tag(self, exp_id: str, k: str, v: str) -> None:
        # Recording isn't necessary for our assertions; track key only.
        if self.raise_on == "set_experiment_tag":
            raise RuntimeError("boom")
        return None

    def search_experiments(self, max_results: int = 1) -> list[FakeExperiment]:
        if self.raise_on == "search_experiments":
            raise RuntimeError("boom")
        return [FakeExperiment(experiment_id=eid, name=n) for n, eid in self.experiments.items()]

    # ---- runs ----
    def create_run(self, experiment_id: str, tags: dict[str, str]) -> FakeRun:
        if self.raise_on == "create_run":
            raise RuntimeError("boom")
        self.next_id += 1
        run_uuid = f"run-{self.next_id:08x}"
        self.runs[run_uuid] = {
            "experiment_id": experiment_id,
            "run_name": tags.get("mlflow.runName"),
            "tags": dict(tags),
            "params": {},
            "metrics": {},
            "status": "RUNNING",
        }
        return FakeRun(info=FakeInfo(run_id=run_uuid))

    def search_runs(
        self,
        experiment_ids: list[str],
        filter_string: str,
        max_results: int = 1,
    ) -> list[FakeRun]:
        # filter_string looks like: tags.mlflow.runName = '<value>'
        if "tags.mlflow.runName = '" not in filter_string:
            return []
        want = filter_string.split("'")[1]
        out: list[FakeRun] = []
        for rid, info in self.runs.items():
            if info["experiment_id"] in experiment_ids and info["run_name"] == want:
                out.append(FakeRun(info=FakeInfo(run_id=rid)))
                if len(out) >= max_results:
                    break
        return out

    def set_tag(self, run_id: str, k: str, v: str) -> None:
        self.runs[run_id]["tags"][k] = v

    def log_param(self, run_id: str, k: str, v: str) -> None:
        if k in self.runs[run_id]["params"]:
            raise RuntimeError("cannot reset existing param")
        self.runs[run_id]["params"][k] = v

    def log_metric(self, run_id: str, k: str, v: float) -> None:
        self.runs[run_id]["metrics"][k] = v

    def set_terminated(self, run_id: str, status: str) -> None:
        self.runs[run_id]["status"] = status


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    """Force MlflowMirror to use a FakeClient and avoid any real mlflow import side-effects."""

    client = FakeClient()
    fake_mlflow = MagicMock()
    fake_mlflow.set_tracking_uri = MagicMock()

    def _build(self: mm.MlflowMirror, *, ping: bool) -> bool:
        self._mlflow = fake_mlflow
        self._client = client  # type: ignore[assignment]
        if ping:
            client.search_experiments(max_results=1)
        return True

    monkeypatch.setattr(mm.MlflowMirror, "_try_build_client", _build)
    return client


# ---------------------------------------------------------------------------
# Disabled-mode behaviour
# ---------------------------------------------------------------------------


def test_disabled_when_url_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LAB_MLFLOW_URL", raising=False)
    # Avoid loading the real settings module (its default is non-empty).
    monkeypatch.setattr(mm.MlflowMirror, "_resolve_uri", staticmethod(lambda _uri=None: ""))
    mirror = mm.MlflowMirror()
    assert mirror.enabled is False
    # all public methods return None and don't raise
    assert mirror.upsert_experiment("s", title="t", plan_path="p", hypothesis=None) is None
    assert mirror.log_run("s", "r", model="m", task="t", seed=0, config={}) is None
    assert mirror.log_finding("F-1", claim="c", importance=1, confidence=0.5) is None
    assert mirror.log_model_card("m", publisher="p", variant=None) is None


def test_disabled_when_ping_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_MLFLOW_URL", "http://broken-host:5000")

    # The production _try_build_client catches exceptions internally and
    # returns False; emulate that path by forcing the inner mlflow import
    # to raise. Real mlflow IS installed, so we have to patch via an
    # ImportError-raising mock at the right level.
    import builtins

    real_import = builtins.__import__

    def _import_raises(name: str, *a: Any, **kw: Any) -> Any:
        if name == "mlflow" or name.startswith("mlflow."):
            raise RuntimeError("nope")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _import_raises)
    mirror = mm.MlflowMirror()
    assert mirror.enabled is False


def test_log_run_noop_on_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mm.MlflowMirror, "_resolve_uri", staticmethod(lambda _uri=None: ""))
    mirror = mm.MlflowMirror()
    assert mirror.enabled is False
    assert mirror.log_run("slug", "run", model="m", task="t", seed=1, config={}) is None


# ---------------------------------------------------------------------------
# Enabled-mode API correctness
# ---------------------------------------------------------------------------


def test_upsert_experiment_creates_and_tags(fake_client: FakeClient) -> None:
    mirror = mm.MlflowMirror(tracking_uri="http://x:5000")
    assert mirror.enabled is True
    eid = mirror.upsert_experiment(
        "EXP-TEST", title="A title", plan_path="docs/exp/x.md", hypothesis="h"
    )
    assert eid is not None
    assert "EXP-TEST" in fake_client.experiments
    # Re-call returns same id (cache + idempotency).
    eid2 = mirror.upsert_experiment(
        "EXP-TEST", title="A title", plan_path="docs/exp/x.md", hypothesis="h"
    )
    assert eid == eid2


def test_log_run_records_params_metrics_tags(fake_client: FakeClient) -> None:
    mirror = mm.MlflowMirror(tracking_uri="http://x:5000")
    rid = mirror.log_run(
        "EXP-A",
        "abc123",
        model="m1",
        task="t1",
        seed=42,
        config={"temperature": 0.5, "max_tokens": 1024},
        params={"system": "you are helpful"},
        metrics={"latency_ms": 250.0, "tokens_in": 30.0},
        tags={"config_hash": "deadbeef"},
        artifact_uri="s3://lab/runs/abc",
    )
    assert rid is not None
    state = fake_client.runs[rid]
    assert state["run_name"] == "abc123"
    assert state["tags"]["lab.experiment_slug"] == "EXP-A"
    assert state["tags"]["lab.model"] == "m1"
    assert state["tags"]["lab.task"] == "t1"
    assert state["tags"]["lab.seed"] == "42"
    assert state["tags"]["config_hash"] == "deadbeef"
    assert state["tags"]["lab.artifact_uri"] == "s3://lab/runs/abc"
    assert state["metrics"]["latency_ms"] == 250.0
    assert state["metrics"]["tokens_in"] == 30.0
    assert state["params"]["system"] == "you are helpful"
    assert state["params"]["config.temperature"] == "0.5"
    assert state["status"] == "FINISHED"


def test_log_run_idempotent_same_run_id_updates_same_mlflow_run(
    fake_client: FakeClient,
) -> None:
    mirror = mm.MlflowMirror(tracking_uri="http://x:5000")
    r1 = mirror.log_run("EXP-A", "abc", model="m", task="t", seed=0, config={})
    # Clear the per-instance run cache so the search_runs path is exercised
    # (covers the "different process / fresh MlflowMirror" idempotency case).
    mirror._run_cache.clear()
    r2 = mirror.log_run(
        "EXP-A",
        "abc",
        model="m",
        task="t",
        seed=0,
        config={},
        metrics={"latency_ms": 999.0},
    )
    assert r1 == r2
    # Only one run created.
    assert len([r for r in fake_client.runs.values() if r["run_name"] == "abc"]) == 1
    assert fake_client.runs[r1]["metrics"]["latency_ms"] == 999.0


def test_log_run_status_failed_propagates(fake_client: FakeClient) -> None:
    mirror = mm.MlflowMirror(tracking_uri="http://x:5000")
    rid = mirror.log_run("EXP-A", "run-x", model="m", task="t", seed=0, config={}, status="FAILED")
    assert rid is not None
    assert fake_client.runs[rid]["status"] == "FAILED"


def test_log_finding_creates_finding_run(fake_client: FakeClient) -> None:
    mirror = mm.MlflowMirror(tracking_uri="http://x:5000")
    rid = mirror.log_finding(
        "F-099", claim="something true", importance=4, confidence=0.6, evidence=["EXP-001"]
    )
    assert rid is not None
    state = fake_client.runs[rid]
    assert state["run_name"] == "F-099"
    assert state["tags"]["lab.finding_slug"] == "F-099"
    assert state["tags"]["lab.confidence"] == "0.6"
    assert state["tags"]["lab.importance"] == "4"
    assert state["metrics"]["importance"] == 4.0
    assert state["metrics"]["confidence"] == pytest.approx(0.6)
    assert state["status"] == "FINISHED"


def test_log_model_card_returns_runs_uri(fake_client: FakeClient) -> None:
    mirror = mm.MlflowMirror(tracking_uri="http://x:5000")
    uri = mirror.log_model_card(
        "model-x",
        publisher="acme",
        variant="8b",
        capabilities=["tool_call", "json"],
        known_issues=["flaky on huge contexts"],
    )
    assert uri is not None
    assert uri.startswith("runs:/")
    rid = uri.split("/")[-1]
    state = fake_client.runs[rid]
    assert state["tags"]["lab.publisher"] == "acme"
    assert state["tags"]["lab.variant"] == "8b"
    assert state["tags"]["lab.capabilities"] == "tool_call,json"


# ---------------------------------------------------------------------------
# Error swallowing — mirror never propagates
# ---------------------------------------------------------------------------


def test_exception_in_log_finding_does_not_propagate(fake_client: FakeClient) -> None:
    fake_client.raise_on = "create_run"
    mirror = mm.MlflowMirror(tracking_uri="http://x:5000")
    # Should not raise; returns None.
    assert mirror.log_finding("F-1", claim="c", importance=1, confidence=0.5) is None


def test_exception_in_upsert_experiment_does_not_propagate(fake_client: FakeClient) -> None:
    fake_client.raise_on = "set_experiment_tag"
    mirror = mm.MlflowMirror(tracking_uri="http://x:5000")
    assert mirror.upsert_experiment("S", title="t", plan_path="p", hypothesis=None) is None


# ---------------------------------------------------------------------------
# is_configured helper
# ---------------------------------------------------------------------------


def test_is_configured_true_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_MLFLOW_URL", "http://localhost:5000")
    assert mm.is_configured() is True


def test_is_configured_false_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LAB_MLFLOW_URL", raising=False)

    class _StubSettings:
        mlflow_url = ""

    def _get_stub() -> _StubSettings:
        return _StubSettings()

    monkeypatch.setattr("lab.core.settings.get_settings", _get_stub)
    assert mm.is_configured() is False
