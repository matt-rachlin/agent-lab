"""External-benchmark adapters (Phase 17.5).

Adapters that turn published benchmark suites (BFCL v3, τ²-bench, …) into
lab Tasks + scorers so they can ride the existing sweep harness. Each
adapter:

  * downloads the raw dataset to a known location under ``~/datasets/``
  * normalises examples into a per-task lab ``Task`` payload
  * exposes a scorer that grades the model's response against the
    benchmark's published ground truth

Adapters are import-light: heavy data only loads when ``register_*`` is
called. This module is safe to import in tests + at lab startup without
materialising the dataset.
"""

from __future__ import annotations

from lab.eval.external import bfcl, bfcl_ast_checker, harbor_suite, tau2

__all__ = ["bfcl", "bfcl_ast_checker", "harbor_suite", "tau2"]
