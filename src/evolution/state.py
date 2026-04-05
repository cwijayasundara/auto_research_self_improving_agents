"""LangGraph state schemas for the evolution pipeline.

Shared types (GraderResult, etc.) now live in evoagent.core.types.
This file only contains LangGraph-specific state TypedDicts.
"""

from typing import Any, TypedDict

from evoagent.core.types import GraderResult
from evoagent.tracing.trajectory import TrajectoryRecord


class AnalysisResult(TypedDict):
    """Aggregated analysis of a single trajectory."""

    run_id: str
    task: str
    classification: str  # "successful", "partial", "failed"
    average_score: float
    grader_results: list[GraderResult]
    output: str
    tool_calls: list[dict[str, Any]]


class AnalyzerState(TypedDict):
    """State for the trajectory analyzer LangGraph workflow."""

    trajectory: TrajectoryRecord
    task_completion: GraderResult
    efficiency: GraderResult
    quality: GraderResult
    claim_verification: GraderResult
    classification: str
    average_score: float


class EvolutionMetrics(TypedDict):
    """Metrics for a single evolution cycle."""

    cycle: int
    avg_score: float
    skills_learned: int
    failure_skills_created: int
    prompt_version: int
    memories_stored: int
    trajectories_analyzed: int


class OrchestratorState(TypedDict):
    """State for the top-level evolution orchestrator."""

    tasks: list[str]
    current_cycle: int
    max_cycles: int
    trajectories: list[TrajectoryRecord]
    analysis_results: list[AnalysisResult]
    cycle_metrics: list[EvolutionMetrics]
    prompt_version: int
    should_continue: bool


class OrchestratorInput(TypedDict):
    """Public input to the orchestrator."""

    tasks: list[str]
    max_cycles: int


class OrchestratorOutput(TypedDict):
    """Public output from the orchestrator."""

    cycle_metrics: list[EvolutionMetrics]
    analysis_results: list[AnalysisResult]
