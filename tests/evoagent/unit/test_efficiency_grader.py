"""Tests for EfficiencyGrader."""

from evoagent.core.types import TrajectoryMetrics
from evoagent.graders.efficiency import EfficiencyGrader


def test_ideal_metrics():
    grader = EfficiencyGrader()
    result = grader.grade(
        task="test", output="test output",
        metrics=TrajectoryMetrics(total_tokens=5000, total_steps=3, latency_seconds=15.0),
    )
    assert result.score > 0.8
    assert result.passed is True
    assert result.name == "efficiency"


def test_bad_metrics():
    grader = EfficiencyGrader()
    result = grader.grade(
        task="test", output="test output",
        metrics=TrajectoryMetrics(total_tokens=100000, total_steps=50, latency_seconds=300.0),
    )
    assert result.score < 0.5
    assert result.passed is False


def test_custom_thresholds():
    grader = EfficiencyGrader(max_ideal_tokens=1000, max_acceptable_tokens=5000)
    result = grader.grade(
        task="test", output="test output",
        metrics=TrajectoryMetrics(total_tokens=3000),
    )
    assert result.score < 1.0
