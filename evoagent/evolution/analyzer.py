"""Trajectory analysis: run graders and classify results.

Accepts any list of Grader implementations. No hardcoded graders.
"""

from __future__ import annotations

import logging
from typing import Any

from evoagent.core.protocols import Grader
from evoagent.core.types import GraderResult, TrajectoryMetrics

logger = logging.getLogger(__name__)

SUCCESSFUL_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.50
MIN_PASS_COUNT = 2
MIN_AVERAGE_SCORE = 0.60

# Graders whose failure caps the classification at "partial" regardless of average.
CRITICAL_GRADERS: set[str] = {"task_completion"}


def classify_trajectory(grader_results: list[GraderResult]) -> tuple[str, float]:
    """Classify a trajectory based on grader results.

    A failing critical grader (e.g. task_completion) caps the result at
    "partial" so the prompt optimizer receives a signal to improve.

    Returns (classification, average_score).
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


def analyze_trajectory(
    graders: list[Grader],
    task: str,
    output: str,
    metrics: TrajectoryMetrics | None = None,
    **kwargs: Any,
) -> list[GraderResult]:
    """Run all graders on a (task, output) pair and return results."""
    if metrics is not None:
        kwargs["metrics"] = metrics

    results: list[GraderResult] = []
    for grader in graders:
        try:
            result = grader.grade(task=task, output=output, **kwargs)
            results.append(result)
        except Exception as exc:
            logger.error("Grader '%s' failed: %s", grader.name, exc)
            results.append(GraderResult(
                name=grader.name, score=0.0, passed=False,
                reasoning=f"grader_error: {exc}",
            ))
    return results
