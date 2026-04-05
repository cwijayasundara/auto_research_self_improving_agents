"""Tests for parallel deep error analysis."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from evoagent.evolution.error_analyzer import analyze_failures_deep, format_error_analysis


def _make_llm_returning(payload: dict) -> MagicMock:
    """Create a mock LLM that returns the given payload as JSON."""
    llm = MagicMock()
    response = MagicMock()
    response.content = json.dumps(payload)
    llm.invoke.return_value = response
    return llm


def _make_failure(task: str = "test task", score: float = 0.2) -> dict:
    return {
        "task": task,
        "output": "Some agent output.",
        "average_score": score,
        "grader_results": [
            {"name": "accuracy", "passed": False, "reasoning": "Wrong answer."}
        ],
    }


def test_returns_analysis_per_failure():
    payload = {
        "root_cause": "missed key facts",
        "counterfactual": "should have searched more",
        "suggested_guidance": "always verify claims",
    }
    llm = _make_llm_returning(payload)
    failures = [_make_failure("task A"), _make_failure("task B")]

    results = analyze_failures_deep(llm, failures)

    assert len(results) == 2
    for result in results:
        assert result["root_cause"] == "missed key facts"
        assert result["counterfactual"] == "should have searched more"
        assert result["suggested_guidance"] == "always verify claims"
    assert results[0]["task"] == "task A"
    assert results[1]["task"] == "task B"


def test_empty_failures_returns_empty():
    llm = MagicMock()

    results = analyze_failures_deep(llm, [])

    assert results == []
    llm.invoke.assert_not_called()


def test_handles_llm_error_gracefully():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("LLM quota exceeded")

    failures = [_make_failure("failing task")]
    results = analyze_failures_deep(llm, failures)

    assert len(results) == 1
    assert results[0]["root_cause"] == "analysis_failed"
    assert results[0]["task"] == "failing task"


def test_caps_at_max_failures():
    payload = {
        "root_cause": "hallucination",
        "counterfactual": "cite sources",
        "suggested_guidance": "double-check facts",
    }
    llm = _make_llm_returning(payload)
    failures = [_make_failure(f"task {i}") for i in range(10)]

    results = analyze_failures_deep(llm, failures, max_failures=5)

    assert len(results) == 5
    # LLM should have been called exactly 5 times
    assert llm.invoke.call_count == 5


def test_format_error_analysis_non_empty():
    analyses = [
        {
            "task": "research quantum computing",
            "root_cause": "insufficient depth",
            "counterfactual": "should use more queries",
            "suggested_guidance": "expand search scope",
        }
    ]
    formatted = format_error_analysis(analyses)

    assert "research quantum computing" in formatted
    assert "insufficient depth" in formatted
    assert "should use more queries" in formatted
    assert "expand search scope" in formatted


def test_format_error_analysis_empty():
    result = format_error_analysis([])
    assert result == "No error analysis available."
