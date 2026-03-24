"""Rule-based grader: efficiency."""

import logging

from src.evolution.state import GraderResult
from src.tracing.trajectory import TrajectoryMetrics

logger = logging.getLogger(__name__)

MAX_IDEAL_TOKENS = 10000
MAX_ACCEPTABLE_TOKENS = 50000
MAX_IDEAL_STEPS = 5
MAX_ACCEPTABLE_STEPS = 15
MAX_IDEAL_LATENCY = 30.0
MAX_ACCEPTABLE_LATENCY = 120.0


def _score_metric(value: float, ideal: float, acceptable: float) -> float:
    """Score a metric on 0-1 scale: ideal=1.0, acceptable=0.5, beyond=0.0."""
    if value <= ideal:
        return 1.0
    if value <= acceptable:
        return 1.0 - 0.5 * ((value - ideal) / (acceptable - ideal))
    return max(0.0, 0.5 - 0.5 * ((value - acceptable) / acceptable))


def grade_efficiency(metrics: TrajectoryMetrics) -> GraderResult:
    """Grade agent efficiency based on resource usage."""
    token_score = _score_metric(metrics.total_tokens, MAX_IDEAL_TOKENS, MAX_ACCEPTABLE_TOKENS)
    step_score = _score_metric(metrics.total_steps, MAX_IDEAL_STEPS, MAX_ACCEPTABLE_STEPS)
    latency_score = _score_metric(
        metrics.latency_seconds, MAX_IDEAL_LATENCY, MAX_ACCEPTABLE_LATENCY
    )

    score = 0.5 * token_score + 0.3 * step_score + 0.2 * latency_score

    reasoning_parts = [
        f"Tokens: {metrics.total_tokens} (score: {token_score:.2f})",
        f"Steps: {metrics.total_steps} (score: {step_score:.2f})",
        f"Latency: {metrics.latency_seconds:.1f}s (score: {latency_score:.2f})",
    ]

    return GraderResult(
        name="efficiency",
        score=round(score, 3),
        passed=score >= 0.5,
        reasoning="; ".join(reasoning_parts),
    )
