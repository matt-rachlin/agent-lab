"""Adapt a lab Task into an Inspect Task. Body lands in 6d."""

from __future__ import annotations

from inspect_ai import Task as InspectTask

from lab.tasks.registry import Task as LabTask


def lab_task_to_inspect(task: LabTask) -> InspectTask:
    raise NotImplementedError("6d — not yet implemented")
