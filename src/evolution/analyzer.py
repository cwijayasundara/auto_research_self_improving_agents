"""Trajectory analyzer LangGraph workflow.

Runs three graders sequentially (task completion, efficiency, quality)
and classifies trajectories as successful, partial, or failed.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from src.evolution.graders.efficiency import grade_efficiency
from src.evolution.graders.quality import grade_quality
from src.evolution.graders.task_completion import grade_task_completion
from src.evolution.state import AnalysisResult, AnalyzerState, GraderResult
from src.tracing.trajectory import Trajectory

logger = logging.getLogger(__name__)

SUCCESSFUL_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.5
MIN_PASS_COUNT = 2
MIN_AVERAGE_SCORE = 0.6


def classify_trajectory(grader_results: list[GraderResult]) -> tuple[str, float]:
    """Classify a trajectory based on grader results."""
    if not grader_results:
        return "failed", 0.0

    scores = [g["score"] for g in grader_results]
    avg_score = sum(scores) / len(scores)
    pass_count = sum(1 for g in grader_results if g["passed"])

    if pass_count >= MIN_PASS_COUNT and avg_score >= MIN_AVERAGE_SCORE:
        if avg_score >= SUCCESSFUL_THRESHOLD:
            return "successful", round(avg_score, 3)
        return "partial", round(avg_score, 3)

    if avg_score >= PARTIAL_THRESHOLD:
        return "partial", round(avg_score, 3)

    return "failed", round(avg_score, 3)


def build_analyzer_graph(llm: BaseChatModel) -> StateGraph:
    """Build the trajectory analyzer as a LangGraph StateGraph."""

    def node_grade_task_completion(state: AnalyzerState) -> dict[str, Any]:
        traj = state["trajectory"]
        result = grade_task_completion(llm, traj.task, traj.output)
        return {"task_completion": result}

    def node_grade_efficiency(state: AnalyzerState) -> dict[str, Any]:
        traj = state["trajectory"]
        result = grade_efficiency(traj.metrics)
        return {"efficiency": result}

    def node_grade_quality(state: AnalyzerState) -> dict[str, Any]:
        traj = state["trajectory"]
        result = grade_quality(llm, traj.task, traj.output)
        return {"quality": result}

    def node_classify(state: AnalyzerState) -> dict[str, Any]:
        grader_results = [
            state["task_completion"],
            state["efficiency"],
            state["quality"],
        ]
        classification, avg_score = classify_trajectory(grader_results)
        return {"classification": classification, "average_score": avg_score}

    graph = StateGraph(AnalyzerState)
    graph.add_node("grade_task_completion", node_grade_task_completion)
    graph.add_node("grade_efficiency", node_grade_efficiency)
    graph.add_node("grade_quality", node_grade_quality)
    graph.add_node("classify", node_classify)

    graph.add_edge(START, "grade_task_completion")
    graph.add_edge("grade_task_completion", "grade_efficiency")
    graph.add_edge("grade_efficiency", "grade_quality")
    graph.add_edge("grade_quality", "classify")
    graph.add_edge("classify", END)

    return graph


def analyze_trajectory(llm: BaseChatModel, trajectory: Trajectory) -> AnalysisResult:
    """Analyze a single trajectory through the grading pipeline."""
    graph = build_analyzer_graph(llm)
    compiled = graph.compile()

    initial_state: AnalyzerState = {
        "trajectory": trajectory,
        "task_completion": GraderResult(name="", score=0, passed=False, reasoning=""),
        "efficiency": GraderResult(name="", score=0, passed=False, reasoning=""),
        "quality": GraderResult(name="", score=0, passed=False, reasoning=""),
        "classification": "",
        "average_score": 0.0,
    }

    result = compiled.invoke(initial_state)

    return AnalysisResult(
        run_id=trajectory.run_id,
        task=trajectory.task,
        classification=result["classification"],
        average_score=result["average_score"],
        grader_results=[
            result["task_completion"],
            result["efficiency"],
            result["quality"],
        ],
        output=trajectory.output,
        tool_calls=[tc.model_dump() for tc in trajectory.tool_calls],
    )
