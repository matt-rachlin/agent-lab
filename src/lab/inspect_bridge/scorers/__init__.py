"""Inspect Scorer subpackage.

The original Phase 6e scorers live one level up at
``lab.inspect_bridge.scorer`` (singular). New families that need their own
helpers — like the RAG scorers added in Phase 6h-c — get their own
module under this package and are re-exported here for ergonomic imports.
"""

from __future__ import annotations

from lab.inspect_bridge.scorers.rag import (
    attribution,
    faithfulness,
    mrr,
    ndcg,
    recall_at_k,
)

__all__ = [
    "attribution",
    "faithfulness",
    "mrr",
    "ndcg",
    "recall_at_k",
]
