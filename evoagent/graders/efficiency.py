"""Rule-based efficiency grader implementing Grader protocol."""

from __future__ import annotations

from evoagent.core.protocols import Grader
from evoagent.core.types import GraderResult, TrajectoryMetrics


class EfficiencyGrader(Grader):
    """Grade agent efficiency based on resource usage."""

    name = "efficiency"

    def __init__(
        self,
        max_ideal_tokens: int = 10_000,
        max_acceptable_tokens: int = 50_000,
        max_ideal_steps: int = 5,
        max_acceptable_steps: int = 15,
        max_ideal_latency: float = 30.0,
        max_acceptable_latency: float = 120.0,
        weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    ) -> None:
        self._ideal_tokens = max_ideal_tokens
        self._accept_tokens = max_acceptable_tokens
        self._ideal_steps = max_ideal_steps
        self._accept_steps = max_acceptable_steps
        self._ideal_latency = max_ideal_latency
        self._accept_latency = max_acceptable_latency
        self._weights = weights

    def grade(self, task: str, output: str, **kwargs) -> GraderResult:
        metrics: TrajectoryMetrics = kwargs.get("metrics", TrajectoryMetrics())

        token_score = _score_metric(metrics.total_tokens, self._ideal_tokens, self._accept_tokens)
        step_score = _score_metric(metrics.total_steps, self._ideal_steps, self._accept_steps)
        latency_score = _score_metric(
            metrics.latency_seconds, self._ideal_latency, self._accept_latency
        )

        w_tok, w_step, w_lat = self._weights
        score = w_tok * token_score + w_step * step_score + w_lat * latency_score

        reasoning = (
            f"Tokens: {metrics.total_tokens} ({token_score:.2f}); "
            f"Steps: {metrics.total_steps} ({step_score:.2f}); "
            f"Latency: {metrics.latency_seconds:.1f}s ({latency_score:.2f})"
        )

        return GraderResult(
            name=self.name,
            score=round(score, 3),
            passed=score >= 0.5,
            reasoning=reasoning,
        )


def _score_metric(value: float, ideal: float, acceptable: float) -> float:
    if value <= ideal:
        return 1.0
    if value <= acceptable:
        return 1.0 - 0.5 * ((value - ideal) / (acceptable - ideal))
    return max(0.0, 0.5 - 0.5 * ((value - acceptable) / acceptable))
