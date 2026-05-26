"""Cross-encoder reranker for stage-2 retrieval.

Wraps sentence-transformers' ``CrossEncoder`` with an idle-TTL unloader so the
reranker model can coexist on a 12 GB GPU with ``qwen3-embedding:8b-q8_0``.

Defaults:
  * Model: ``Qwen/Qwen3-Reranker-0.6B`` (Apache 2.0, ~1.2 GB VRAM, MTEB-R 61.82).
  * Fallback: ``BAAI/bge-reranker-v2-m3`` (Apache 2.0, BEIR 56.51, more mature).
  * Idle unload: 300 s without a call triggers a release of model + tokenizer.

Selection / disable:
  * Env var ``LAB_RAG_RERANKER`` overrides the constructor's ``model_name``.
  * ``LAB_RAG_RERANKER=none`` short-circuits — :meth:`LabReranker.rerank`
    returns the input list unchanged (clamped to ``top_n``).

Threading: single-threaded. The first call loads the model; subsequent calls
reuse it until :meth:`unload` is called or the idle reaper fires.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

from lab.rag import DEFAULT_RERANKER_MODEL, FALLBACK_RERANKER_MODEL, RERANKER_ENV_VAR

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

#: Sentinel value for "no reranker — pass-through". When the env var or
#: constructor receives this, :meth:`LabReranker.rerank` clamps to ``top_n``
#: but does not load a model.
RERANKER_DISABLED = "none"


class LabReranker:
    """Cross-encoder wrapper with lazy load + idle unload.

    Args:
        model_name: HuggingFace model id (e.g. ``"Qwen/Qwen3-Reranker-0.6B"``).
            If ``None``, reads ``LAB_RAG_RERANKER`` from the environment, then
            falls back to :data:`lab.rag.DEFAULT_RERANKER_MODEL`.
        idle_unload_sec: Idle window before the model is released. Set to 0
            (or negative) to disable the reaper — useful for tests that never
            want to wait for it to fire.
    """

    def __init__(
        self,
        model_name: str | None = None,
        idle_unload_sec: int = 300,
    ) -> None:
        env_choice = os.environ.get(RERANKER_ENV_VAR, "").strip()
        chosen = (model_name or env_choice or DEFAULT_RERANKER_MODEL).strip()
        if chosen == "":
            chosen = DEFAULT_RERANKER_MODEL
        self.model_name: str = chosen
        self.idle_unload_sec: int = int(idle_unload_sec)
        self._model: CrossEncoder | None = None
        self._last_used: float = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # public surface
    # ------------------------------------------------------------------

    @property
    def disabled(self) -> bool:
        """True iff the reranker is configured to pass-through (no model load)."""
        return self.model_name == RERANKER_DISABLED

    @property
    def loaded(self) -> bool:
        """True iff the cross-encoder weights are currently resident."""
        return self._model is not None

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Re-score ``candidates`` against ``query`` and return the top ``top_n``.

        Each candidate must carry a ``"text"`` field; the reranker scores the
        (query, text) pair and the returned dicts gain a ``"rerank_score"``
        float. Original ordering is preserved as a tie-breaker.

        When :attr:`disabled` is True, returns ``candidates[:top_n]`` unchanged
        (no model load, no score injection).
        """
        if top_n <= 0:
            return []
        if not candidates:
            return []
        if self.disabled:
            return list(candidates[:top_n])

        texts = [str(c.get("text", "")) for c in candidates]
        pairs = [(query, t) for t in texts]

        self._maybe_unload()
        model = self._ensure_loaded()
        self._last_used = time.monotonic()

        # CrossEncoder.predict returns one float per pair.
        scores = list(model.predict(pairs))
        scored: list[tuple[int, float, dict[str, Any]]] = []
        for idx, (cand, raw) in enumerate(zip(candidates, scores, strict=True)):
            out = dict(cand)
            out["rerank_score"] = float(raw)
            scored.append((idx, float(raw), out))

        # Stable sort: rerank_score desc, original index asc for ties.
        scored.sort(key=lambda t: (-t[1], t[0]))
        return [t[2] for t in scored[:top_n]]

    def unload(self) -> None:
        """Release the model + tokenizer so VRAM is freed.

        Idempotent — calling ``unload`` when no model is loaded is a no-op.
        """
        with self._lock:
            if self._model is None:
                return
            try:
                # Best-effort: drop refs to torch tensors so the next gc cycle
                # (or torch.cuda.empty_cache) reclaims VRAM.
                model = self._model
                self._model = None
                del model
            except Exception:  # pragma: no cover - defensive
                self._model = None
            self._maybe_empty_cuda_cache()
            logger.info("LabReranker unloaded %s", self.model_name)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> CrossEncoder:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - import guard
                raise RuntimeError(
                    "sentence-transformers not installed; required for the "
                    "cross-encoder reranker. Install via "
                    "`uv pip install sentence-transformers` or set "
                    f"{RERANKER_ENV_VAR}={RERANKER_DISABLED}."
                ) from exc
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                if self.model_name != FALLBACK_RERANKER_MODEL:
                    logger.warning(
                        "primary reranker %s failed to load (%s); falling back to %s",
                        self.model_name,
                        exc,
                        FALLBACK_RERANKER_MODEL,
                    )
                    self.model_name = FALLBACK_RERANKER_MODEL
                    self._model = CrossEncoder(FALLBACK_RERANKER_MODEL)
                else:
                    raise
            logger.info("LabReranker loaded %s", self.model_name)
            return self._model

    def _maybe_unload(self) -> None:
        if self.idle_unload_sec <= 0 or self._model is None:
            return
        if (time.monotonic() - self._last_used) >= self.idle_unload_sec:
            self.unload()

    @staticmethod
    def _maybe_empty_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():  # pragma: no cover - CUDA path
                torch.cuda.empty_cache()
        except Exception:
            return


# Module-level singleton: callers can share one instance to avoid loading the
# cross-encoder twice. Construct lazily so importing the module is cheap.
_DEFAULT_RERANKER: LabReranker | None = None
_DEFAULT_LOCK = threading.Lock()


def get_default_reranker() -> LabReranker:
    """Process-wide singleton :class:`LabReranker`. Lazy on first call."""
    global _DEFAULT_RERANKER  # noqa: PLW0603
    if _DEFAULT_RERANKER is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_RERANKER is None:
                _DEFAULT_RERANKER = LabReranker()
    return _DEFAULT_RERANKER


def reset_default_reranker() -> None:
    """Clear the singleton (mostly for tests)."""
    global _DEFAULT_RERANKER  # noqa: PLW0603
    with _DEFAULT_LOCK:
        if _DEFAULT_RERANKER is not None:
            _DEFAULT_RERANKER.unload()
        _DEFAULT_RERANKER = None
