"""Tests for evoagent core types."""

from evoagent.core.types import GraderResult, TaskResult, TrajectoryMetrics, EvolutionCycleReport


def test_grader_result_fields():
    r = GraderResult(name="quality", score=0.85, passed=True, reasoning="Good structure")
    assert r.name == "quality"
    assert r.score == 0.85
    assert r.passed is True
    assert r.reasoning == "Good structure"


def test_task_result_fields():
    r = TaskResult(
        task="Research quantum computing",
        output="Report content...",
        tool_calls=[{"name": "search", "args": {"q": "quantum"}}],
        duration_seconds=25.0,
        status="success",
    )
    assert r.status == "success"
    assert r.duration_seconds == 25.0


def test_trajectory_metrics_defaults():
    m = TrajectoryMetrics()
    assert m.total_tokens == 0
    assert m.total_steps == 0
    assert m.latency_seconds == 0.0
    assert m.tool_call_count == 0


def test_evolution_cycle_report():
    r = EvolutionCycleReport(
        cycle=3,
        avg_score=0.82,
        classification_counts={"successful": 2, "partial": 1, "failed": 0},
        prompt_version=5,
        skills_created=2,
        improvements=["Better error handling"],
    )
    assert r.cycle == 3
    assert r.avg_score == 0.82
