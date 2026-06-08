"""
Prism Task Graph Engine.

Source: Proximus orchestrator/task_graph.py (simplified)

Provides a DAG-based orchestration engine using NetworkX.
Enables parallel execution of review agents while respecting dependencies.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
import uuid

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field
from observability.logging import get_logger

logger = get_logger(__name__)


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Task(BaseModel):
    """Represents a single step in the review pipeline."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    agent_role: str  # ContextFetcher, Security, Quality, Performance, Aggregator
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = []
    input_data: dict[str, Any] = {}
    output_data: dict[str, Any] | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = ConfigDict(use_enum_values=True)

    def mark_started(self):
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now(UTC)

    def mark_completed(self, output: dict[str, Any]):
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now(UTC)
        self.output_data = output

    def mark_failed(self, error: str):
        self.status = TaskStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.now(UTC)


class TaskGraph:
    """
    DAG-based orchestration for the Prism review pipeline.
    """

    def __init__(self, review_id: str):
        self.review_id = review_id
        self.graph = nx.DiGraph()
        self.tasks: dict[str, Task] = {}

    def add_task(self, task: Task) -> str:
        """Add a task to the DAG."""
        self.tasks[task.id] = task
        self.graph.add_node(task.id, task=task)

        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                raise ValueError(f"Dependency '{dep_id}' not found")
            self.graph.add_edge(dep_id, task.id)

        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_node(task.id)
            del self.tasks[task.id]
            raise ValueError(f"Task '{task.name}' creates a cycle")

        return task.id

    def get_ready_tasks(self) -> list[Task]:
        """Tasks whose dependencies have all resolved (COMPLETED or FAILED).

        A task becomes ready even if some deps failed — the executor
        decides whether to run or skip it based on its own logic.
        """
        ready = []
        for task_id, task in self.tasks.items():
            if task.status != TaskStatus.PENDING:
                continue
            deps = list(self.graph.predecessors(task_id))
            all_resolved = all(
                self.tasks[dep].status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED) for dep in deps
            )
            if all_resolved:
                ready.append(task)
        return ready

    def has_failed_dependency(self, task: Task) -> bool:
        """Check if any of this task's upstream dependencies failed or were skipped."""
        deps = list(self.graph.predecessors(task.id))
        return any(self.tasks[dep].status in (TaskStatus.FAILED, TaskStatus.SKIPPED) for dep in deps)

    def is_complete(self) -> bool:
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED) for t in self.tasks.values()
        )

    def get_status_summary(self) -> dict[str, Any]:
        counts = {s: 0 for s in TaskStatus}
        for t in self.tasks.values():
            counts[t.status] += 1
        return {
            "total": len(self.tasks),
            **counts,
            "progress": round((counts[TaskStatus.COMPLETED] / len(self.tasks)) * 100, 1) if self.tasks else 0,
        }
