"""Trajectory analyzer LangGraph workflow.

Runs four graders in parallel (task completion, efficiency, quality,
claim verification) and classifies trajectories as successful, partial,
or failed.
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from evoagent.core.types import GraderResult, TrajectoryMetrics
from evoagent.graders.efficiency import EfficiencyGrader
from evoagent.graders.multi_judge import MultiJudgeGrader
from evoagent.tracing.trajectory import TrajectoryRecord

from src.agent.prompts import (
    Q_DEPTH_PROMPT,
    Q_RELEVANCE_PROMPT,
    Q_STRUCTURE_PROMPT,
    QUALITY_PROMPT,
    TC_ACCURACY_PROMPT,
    TC_COMPLETENESS_PROMPT,
    TC_EVIDENCE_PROMPT,
    TASK_COMPLETION_PROMPT,
)
from src.evolution.graders.claim_verification import _extract_claims, grade_claims
from src.evolution.graders.fact_checker import spot_check_claims
from src.evolution.state import AnalysisResult, AnalyzerState

logger = logging.getLogger(__name__)

SUCCESSFUL_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.5
MIN_PASS_COUNT = 2
MIN_AVERAGE_SCORE = 0.6

# Graders whose failure caps the classification at "partial" regardless of average.
CRITICAL_GRADERS = {"task_completion"}


def classify_trajectory(grader_results: list[GraderResult]) -> tuple[str, float]:
    """Classify a trajectory based on grader results.

    A failing critical grader (e.g. task_completion) caps the result at
    "partial" so the prompt optimizer receives a signal to improve.
    """
    if not grader_results:
        return "failed", 0.0

    scores = [g.score for g in grader_results]
    avg_score = sum(scores) / len(scores)
    pass_count = sum(1 for g in grader_results if g.passed)

    critical_failed = any(
        not g.passed for g in grader_results if g.name in CRITICAL_GRADERS
    )

    if pass_count >= MIN_PASS_COUNT and avg_score >= MIN_AVERAGE_SCORE:
        if avg_score >= SUCCESSFUL_THRESHOLD and not critical_failed:
            return "successful", round(avg_score, 3)
        return "partial", round(avg_score, 3)

    if avg_score >= PARTIAL_THRESHOLD:
        return "partial", round(avg_score, 3)

    return "failed", round(avg_score, 3)


def _trajectory_to_metrics(trajectory: TrajectoryRecord) -> TrajectoryMetrics:
    """Convert TrajectoryRecord flat fields to TrajectoryMetrics."""
    return TrajectoryMetrics(
        total_tokens=trajectory.total_tokens,
        total_steps=trajectory.total_steps,
        latency_seconds=trajectory.latency_seconds,
        tool_call_count=trajectory.tool_call_count,
    )


def build_analyzer_graph(llm: BaseChatModel, search_tool: BaseTool | None = None) -> StateGraph:
    """Build the trajectory analyzer as a LangGraph StateGraph.

    The four graders run in parallel from START, then converge on classify.
    """
    task_completion_grader = MultiJudgeGrader(
        llm=llm,
        name="task_completion",
        judge_prompts=[TC_COMPLETENESS_PROMPT, TC_EVIDENCE_PROMPT, TC_ACCURACY_PROMPT],
        fallback_prompt=TASK_COMPLETION_PROMPT,
    )

    quality_grader = MultiJudgeGrader(
        llm=llm,
        name="quality",
        judge_prompts=[Q_STRUCTURE_PROMPT, Q_DEPTH_PROMPT, Q_RELEVANCE_PROMPT],
        fallback_prompt=QUALITY_PROMPT,
    )

    efficiency_grader = EfficiencyGrader()

    def node_grade_task_completion(state: AnalyzerState) -> dict[str, Any]:
        traj = state["trajectory"]
        result = task_completion_grader.grade(traj.task, traj.output)
        return {"task_completion": result}

    def node_grade_efficiency(state: AnalyzerState) -> dict[str, Any]:
        traj = state["trajectory"]
        metrics = _trajectory_to_metrics(traj)
        result = efficiency_grader.grade(traj.task, traj.output, metrics=metrics)
        return {"efficiency": result}

    def node_grade_quality(state: AnalyzerState) -> dict[str, Any]:
        traj = state["trajectory"]
        result = quality_grader.grade(traj.task, traj.output)
        return {"quality": result}

    def node_grade_claims(state: AnalyzerState) -> dict[str, Any]:
        traj = state["trajectory"]
        claims = _extract_claims(llm, traj.output)
        spot_score = None
        if search_tool and claims:
            spot_score = spot_check_claims(llm, search_tool, claims)
        result = grade_claims(llm, traj.task, traj.output, spot_check_score=spot_score)
        return {"claim_verification": result}

    def node_classify(state: AnalyzerState) -> dict[str, Any]:
        grader_results = [
            state["task_completion"],
            state["efficiency"],
            state["quality"],
            state["claim_verification"],
        ]
        classification, avg_score = classify_trajectory(grader_results)
        return {"classification": classification, "average_score": avg_score}

    graph = StateGraph(AnalyzerState)
    graph.add_node("grade_task_completion", node_grade_task_completion)
    graph.add_node("grade_efficiency", node_grade_efficiency)
    graph.add_node("grade_quality", node_grade_quality)
    graph.add_node("grade_claims", node_grade_claims)
    graph.add_node("classify", node_classify)

    # Parallel fan-out from START to all 4 graders
    graph.add_edge(START, "grade_task_completion")
    graph.add_edge(START, "grade_efficiency")
    graph.add_edge(START, "grade_quality")
    graph.add_edge(START, "grade_claims")

    # Fan-in: all 4 graders converge on classify
    graph.add_edge("grade_task_completion", "classify")
    graph.add_edge("grade_efficiency", "classify")
    graph.add_edge("grade_quality", "classify")
    graph.add_edge("grade_claims", "classify")

    graph.add_edge("classify", END)

    return graph


def analyze_trajectory(
    llm: BaseChatModel,
    trajectory: TrajectoryRecord,
    search_tool: BaseTool | None = None,
) -> AnalysisResult:
    """Analyze a single trajectory through the grading pipeline."""
    graph = build_analyzer_graph(llm, search_tool=search_tool)
    compiled = graph.compile()

    initial_state: AnalyzerState = {
        "trajectory": trajectory,
        "task_completion": GraderResult(name="", score=0, passed=False, reasoning=""),
        "efficiency": GraderResult(name="", score=0, passed=False, reasoning=""),
        "quality": GraderResult(name="", score=0, passed=False, reasoning=""),
        "claim_verification": GraderResult(name="", score=0, passed=False, reasoning=""),
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
            result["claim_verification"],
        ],
        output=trajectory.output,
        tool_calls=[tc.model_dump() for tc in trajectory.tool_calls],
    )
