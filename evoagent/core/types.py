"""Shared types for the evoagent framework.

All types are plain dataclasses with no heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GraderResult:
    """Result from a single grader evaluation."""

    name: str
    score: float  # 0.0 - 1.0
    passed: bool
    reasoning: str


@dataclass
class TaskResult:
    """Result from running the agent on a single task."""

    task: str
    output: str
    tool_calls: list[dict] = field(default_factory=list)
    duration_seconds: float = 0.0
    status: Literal["success", "error", "timeout"] = "success"


@dataclass
class TrajectoryMetrics:
    """Efficiency metrics for a single agent run."""

    total_tokens: int = 0
    total_steps: int = 0
    latency_seconds: float = 0.0
    tool_call_count: int = 0


@dataclass
class EvolutionCycleReport:
    """Summary of a single evolution cycle."""

    cycle: int
    avg_score: float
    classification_counts: dict[str, int] = field(default_factory=dict)
    prompt_version: int = 0
    skills_created: int = 0
    improvements: list[str] = field(default_factory=list)
