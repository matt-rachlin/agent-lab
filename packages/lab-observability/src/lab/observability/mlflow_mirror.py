"""Best-effort additive mirror of lab operational state into MLflow.

Postgres remains the canonical source of truth for experiments, runs,
findings and the model registry. This module mirrors a subset of those
writes into MLflow so the visual artifact browser, run comparison and
sweep dashboards work without a migration.

Every public method is:

* idempotent on the lab-side identity (experiment slug / run id / finding
  slug / litellm_id) — re-calling with the same key updates the same
  MLflow record rather than creating a duplicate
* fire-and-forget — any exception inside the mirror is caught and logged;
  the canonical Postgres write is never blocked
* a no-op when MLflow is unreachable or `LAB_MLFLOW_URL` is unset

Design notes
------------

We deliberately do NOT import ``mlflow`` at module-import time. The
constructor pulls it in only when the mirror is enabled, so unit tests
that monkey-patch the loader can run without the heavy dependency, and
the import cost is paid lazily by code paths that actually mirror.

The mirror caches the connectivity-check result for the lifetime of the
process: once we decide the server is down we stop trying for the rest of
the session. A long-running sweep that loses the MLflow server halfway
through will silently degrade — the Postgres rows still land.
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from urllib.parse import urlparse

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mlflow.tracking import MlflowClient as _MlflowClient


RunStatus = Literal["FINISHED", "FAILED"]


def _log(message: str) -> None:
    """Stderr log helper. Lab hasn't adopted structlog yet so we keep it light."""

    print(f"[mlflow_mirror] {message}", file=sys.stderr)


class MlflowMirror:
    """Fire-and-forget mirror of lab state into MLflow tracking.

    Parameters
    ----------
    tracking_uri:
        MLflow tracking URI. When None we read ``LAB_MLFLOW_URL`` from the
        environment; if that's also unset we fall back to
        ``lab.core.settings.get_settings().mlflow_url``.
    enabled:
        Force the mirror on/off. When None (the default) the mirror
        decides based on URI + ping result.
    """

    # Class-level singleton-ish cache: many call sites construct a fresh
    # ``MlflowMirror()`` per operation; we'd rather not pay the import +
    # ping cost each time. The cache is keyed on the resolved tracking URI
    # so tests that monkey-patch the env still get a fresh instance.
    _cached: ClassVar[dict[tuple[str, bool | None], MlflowMirror]] = {}

    def __init__(
        self,
        tracking_uri: str | None = None,
        *,
        enabled: bool | None = None,
    ) -> None:
        self._tracking_uri = self._resolve_uri(tracking_uri)
        self._client: _MlflowClient | None = None
        self._mlflow: Any | None = None
        # Experiment id cache: slug -> mlflow experiment id (decimal string).
        self._exp_cache: dict[str, str] = {}
        # Run id cache: lab run_id -> mlflow run uuid.
        self._run_cache: dict[str, str] = {}

        if enabled is False:
            self.enabled = False
            return

        if not self._tracking_uri:
            self.enabled = False
            return

        # When the caller explicitly forces enabled=True we still try to
        # build a client but skip the ping; this is what the tests use.
        if enabled is True:
            self.enabled = self._try_build_client(ping=False)
            return

        self.enabled = self._try_build_client(ping=True)

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_uri(tracking_uri: str | None) -> str:
        if tracking_uri:
            return tracking_uri
        env = os.environ.get("LAB_MLFLOW_URL")
        if env:
            return env
        try:
            from lab.core.settings import get_settings

            return get_settings().mlflow_url or ""
        except Exception:
            return ""

    def _try_build_client(self, *, ping: bool) -> bool:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(self._tracking_uri)
            self._mlflow = mlflow
            self._client = MlflowClient(tracking_uri=self._tracking_uri)
            if ping:
                # Cheapest reliable round-trip: list experiments with max_results=1.
                self._client.search_experiments(max_results=1)
        except Exception as exc:
            _log(f"disabled (init/ping failed: {type(exc).__name__}: {exc})")
            self._client = None
            self._mlflow = None
            return False
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_experiment(
        self,
        slug: str,
        *,
        title: str,
        plan_path: str,
        hypothesis: str | None,
    ) -> str | None:
        """Create-or-touch the MLflow experiment for `slug`. Returns the MLflow id."""

        if not self.enabled or self._client is None:
            return None
        try:
            exp_id = self._get_or_create_experiment(slug)
            tags = {
                "lab.slug": slug,
                "lab.title": title,
                "lab.plan_path": plan_path,
            }
            if hypothesis:
                tags["lab.hypothesis"] = hypothesis
            for k, v in tags.items():
                self._client.set_experiment_tag(exp_id, k, v)
            self._exp_cache[slug] = exp_id
            return exp_id
        except Exception as exc:
            _log(f"upsert_experiment({slug}) failed: {type(exc).__name__}: {exc}")
            return None

    def log_run(
        self,
        experiment_slug: str,
        run_id: str,
        *,
        model: str,
        task: str,
        seed: int,
        config: dict[str, Any],
        params: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
        artifact_uri: str | None = None,
        status: RunStatus = "FINISHED",
    ) -> str | None:
        """Mirror one experiment cell into MLflow. Returns the MLflow run uuid."""

        if not self.enabled or self._client is None or self._mlflow is None:
            return None
        try:
            exp_id = self._get_or_create_experiment(experiment_slug)
            # Find an existing MLflow run for this lab run_id so re-runs
            # update in place rather than spawning duplicates.
            existing = self._find_run_by_name(exp_id, run_id)
            if existing:
                mlflow_run_id = existing
            else:
                run = self._client.create_run(
                    experiment_id=exp_id,
                    tags={"mlflow.runName": run_id},
                )
                mlflow_run_id = str(run.info.run_id)

            base_tags: dict[str, str] = {
                "lab.experiment_slug": experiment_slug,
                "lab.run_id": run_id,
                "lab.model": model,
                "lab.task": task,
                "lab.seed": str(seed),
            }
            if tags:
                for k, v in tags.items():
                    if v is not None:
                        base_tags[k] = str(v)
            for k, v in base_tags.items():
                self._client.set_tag(mlflow_run_id, k, v)

            # Params: stable string dump of the config plus any extras.
            param_dump: dict[str, str] = {}
            for k, v in (params or {}).items():
                param_dump[str(k)[:250]] = _stringify(v)
            # Flatten a small subset of config into top-level params for
            # quick UI filtering — the full config still goes in tags as
            # a JSON blob.
            for k in ("temperature", "top_p", "max_tokens"):
                if k in config and config[k] is not None:
                    param_dump.setdefault(f"config.{k}", _stringify(config[k]))
            if config:
                import json

                base_tags["lab.config_json"] = json.dumps(config, default=str)[:5000]
                self._client.set_tag(mlflow_run_id, "lab.config_json", base_tags["lab.config_json"])
            for k, v in param_dump.items():
                # MLflow rejects re-setting params; tolerate the duplicate.
                with contextlib.suppress(Exception):
                    self._client.log_param(mlflow_run_id, k, v)

            for metric_name, metric_value in (metrics or {}).items():
                if metric_value is None:
                    continue
                with contextlib.suppress(TypeError, ValueError):
                    self._client.log_metric(mlflow_run_id, metric_name, float(metric_value))

            if artifact_uri:
                self._client.set_tag(mlflow_run_id, "lab.artifact_uri", artifact_uri)

            mlflow_status = "FINISHED" if status == "FINISHED" else "FAILED"
            self._client.set_terminated(mlflow_run_id, status=mlflow_status)
            self._run_cache[run_id] = mlflow_run_id
            return mlflow_run_id
        except Exception as exc:
            _log(f"log_run({run_id}) failed: {type(exc).__name__}: {exc}")
            return None

    def log_finding(
        self,
        finding_id: str,
        *,
        claim: str,
        importance: int,
        confidence: float,
        evidence: list[str] | None = None,
    ) -> str | None:
        """Mirror a finding into a dedicated MLflow experiment called ``lab-findings``."""

        if not self.enabled or self._client is None:
            return None
        try:
            exp_id = self._get_or_create_experiment("lab-findings")
            existing = self._find_run_by_name(exp_id, finding_id)
            if existing:
                mlflow_run_id = existing
            else:
                run = self._client.create_run(
                    experiment_id=exp_id,
                    tags={"mlflow.runName": finding_id},
                )
                mlflow_run_id = str(run.info.run_id)

            tags = {
                "lab.finding_slug": finding_id,
                "lab.claim": claim[:500],
                "lab.confidence": str(confidence),
                "lab.importance": str(importance),
            }
            for k, v in tags.items():
                self._client.set_tag(mlflow_run_id, k, v)
            with contextlib.suppress(TypeError, ValueError):
                self._client.log_metric(mlflow_run_id, "importance", float(importance))
            with contextlib.suppress(TypeError, ValueError):
                self._client.log_metric(mlflow_run_id, "confidence", float(confidence))
            if evidence:
                self._client.set_tag(
                    mlflow_run_id,
                    "lab.evidence",
                    "; ".join(str(e) for e in evidence)[:5000],
                )
            self._client.set_terminated(mlflow_run_id, status="FINISHED")
            return mlflow_run_id
        except Exception as exc:
            _log(f"log_finding({finding_id}) failed: {type(exc).__name__}: {exc}")
            return None

    def log_model_card(
        self,
        litellm_id: str,
        *,
        publisher: str,
        variant: str | None,
        capabilities: list[str] | None = None,
        known_issues: list[str] | None = None,
    ) -> str | None:
        """Mirror a model registry entry. Returns the MLflow model URI (or run id fallback)."""

        if not self.enabled or self._client is None:
            return None
        try:
            exp_id = self._get_or_create_experiment("lab-models")
            existing = self._find_run_by_name(exp_id, litellm_id)
            if existing:
                mlflow_run_id = existing
            else:
                run = self._client.create_run(
                    experiment_id=exp_id,
                    tags={"mlflow.runName": litellm_id},
                )
                mlflow_run_id = str(run.info.run_id)

            tags = {
                "lab.litellm_id": litellm_id,
                "lab.publisher": publisher,
            }
            if variant:
                tags["lab.variant"] = variant
            if capabilities:
                tags["lab.capabilities"] = ",".join(capabilities)
            if known_issues:
                tags["lab.known_issues"] = "; ".join(known_issues)[:5000]
            for k, v in tags.items():
                self._client.set_tag(mlflow_run_id, k, v)
            self._client.set_terminated(mlflow_run_id, status="FINISHED")
            # We store the mlflow run id as the canonical mirror handle —
            # mlflow_model_uri is a stable identifier even if no Model
            # Registry entry exists yet. Format: `runs:/<run_id>`.
            return f"runs:/{mlflow_run_id}"
        except Exception as exc:
            _log(f"log_model_card({litellm_id}) failed: {type(exc).__name__}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_experiment(self, name: str) -> str:
        cached = self._exp_cache.get(name)
        if cached:
            return cached
        assert self._client is not None
        exp = self._client.get_experiment_by_name(name)
        if exp is not None:
            exp_id = str(exp.experiment_id)
            # If the experiment was soft-deleted (e.g. by a previous smoke
            # test) we restore it; otherwise create_run would fail with
            # INVALID_PARAMETER_VALUE: "must be in the 'active' state".
            if getattr(exp, "lifecycle_stage", "active") == "deleted":
                with contextlib.suppress(Exception):
                    self._client.restore_experiment(exp_id)
        else:
            exp_id = str(self._client.create_experiment(name))
        self._exp_cache[name] = exp_id
        return exp_id

    def _find_run_by_name(self, exp_id: str, run_name: str) -> str | None:
        cached = self._run_cache.get(run_name)
        if cached:
            return cached
        assert self._client is not None
        # MLflow's search uses `tags.mlflow.runName` to filter by run name.
        results = self._client.search_runs(
            experiment_ids=[exp_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            max_results=1,
        )
        if results:
            run_id = str(results[0].info.run_id)
            self._run_cache[run_name] = run_id
            return run_id
        return None


def _stringify(v: Any) -> str:
    """MLflow params must be strings ≤ 6000 chars. Truncate aggressively."""

    if v is None:
        return ""
    s = str(v)
    if len(s) > 250:
        s = s[:247] + "..."
    return s


def is_configured() -> bool:
    """Cheap check: does *something* point us at MLflow?"""

    if os.environ.get("LAB_MLFLOW_URL"):
        return True
    try:
        from lab.core.settings import get_settings

        url = get_settings().mlflow_url or ""
    except Exception:
        return False
    if not url:
        return False
    parsed = urlparse(url)
    return bool(parsed.scheme and parsed.netloc)


__all__ = ["MlflowMirror", "RunStatus", "is_configured"]
