"""Tests for trajectory analyzer."""

from evoagent.core.types import GraderResult, TrajectoryMetrics
from evoagent.evolution.analyzer import analyze_trajectory, classify_trajectory
from evoagent.graders.efficiency import EfficiencyGrader


def test_classify_successful():
    results = [
        GraderResult(name="a", score=0.9, passed=True, reasoning="good"),
        GraderResult(name="b", score=0.8, passed=True, reasoning="good"),
        GraderResult(name="c", score=0.7, passed=False, reasoning="ok"),
    ]
    classification, avg = classify_trajectory(results)
    assert classification == "successful"
    assert avg > 0.75


def test_classify_failed():
    results = [
        GraderResult(name="a", score=0.2, passed=False, reasoning="bad"),
        GraderResult(name="b", score=0.3, passed=False, reasoning="bad"),
    ]
    classification, _avg = classify_trajectory(results)
    assert classification == "failed"


def test_classify_empty():
    classification, _avg = classify_trajectory([])
    assert classification == "failed"
    assert _avg == 0.0


def test_analyze_with_graders():
    graders = [EfficiencyGrader()]
    results = analyze_trajectory(
        graders=graders,
        task="test task",
        output="test output",
        metrics=TrajectoryMetrics(total_tokens=5000, total_steps=3, latency_seconds=15.0),
    )
    assert len(results) == 1
    assert results[0].name == "efficiency"
