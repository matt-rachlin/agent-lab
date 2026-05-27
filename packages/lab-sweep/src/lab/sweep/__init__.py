"""Sweep package — comparison-sweep orchestration."""

from lab.sweep.config import RunConfig, SweepConfig, config_hash, load_sweep, run_id
from lab.sweep.runner import Cell, CellResult, execute_cell, expand_matrix, run_sweep

__all__ = [
    "Cell",
    "CellResult",
    "RunConfig",
    "SweepConfig",
    "config_hash",
    "execute_cell",
    "expand_matrix",
    "load_sweep",
    "run_id",
    "run_sweep",
]
