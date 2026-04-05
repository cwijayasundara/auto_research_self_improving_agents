"""Tests for harness optimizer module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.evolution.harness_config import HarnessConfig, HarnessConfigStore
from src.evolution.harness_optimizer import (
    build_harness_diagnosis,
    propose_harness_changes,
    optimize_harness,
)
from src.evolution.run_log import RunLogEntry


def _make_entry(
    score: float = 0.5,
    classification: str = "research",
    grader_results: list[dict] | None = None,
    trace_path: str = "",
) -> RunLogEntry:
    entry = RunLogEntry(
        run_id="test-001",
        task="What is quantum computing?",
        output="Some output about quantum computing.",
        classification=classification,
        average_score=score,
        grader_results=grader_results
        or [
            {"dimension": "completeness", "score": score, "reason": "Too short"},
            {"dimension": "accuracy", "score": score + 0.1, "reason": "OK"},
        ],
        prompt_version=1,
    )
    # Task 3 may not have added these fields yet; attach manually
    entry.harness_config_version = 0
    entry.trace_path = trace_path
    return entry


class TestBuildHarnessDiagnosis:
    def test_includes_grader_failure_info_and_config(self):
        entries = [
            _make_entry(score=0.3),
            _make_entry(score=0.4),
        ]
        config = HarnessConfig(max_retries=3, min_length=800)
        diagnosis = build_harness_diagnosis(entries, config)

        # Should include dimension names from grader results
        assert "completeness" in diagnosis
        assert "accuracy" in diagnosis
        # Should include current config values
        assert "max_retries" in diagnosis
        assert "3" in diagnosis
        assert "min_length" in diagnosis
        assert "800" in diagnosis
        # Should include average score info
        assert "0.3" in diagnosis or "0.35" in diagnosis


class TestProposeHarnessChanges:
    def test_returns_valid_config(self):
        llm = MagicMock()
        llm.return_value = json.dumps(
            {
                "changes": {
                    "max_retries": 4,
                    "min_length": 1000,
                    "planning_effort": "high",
                },
                "reasoning": "Increasing retries to improve reliability.",
            }
        )

        config = HarnessConfig()
        new_config, reasoning = propose_harness_changes(
            llm, "some diagnosis", config
        )

        assert new_config.max_retries == 4
        assert new_config.min_length == 1000
        assert new_config.planning_effort == "high"
        assert "reliability" in reasoning.lower() or len(reasoning) > 0

    def test_clamps_extreme_values(self):
        llm = MagicMock()
        llm.return_value = json.dumps(
            {
                "changes": {
                    "max_retries": 999,
                    "min_length": 1,
                    "max_similar": 0,
                    "budget_seconds": 9999,
                    "planning_calls": -5,
                    "planning_effort": "extreme",  # invalid
                },
                "reasoning": "Push limits.",
            }
        )

        config = HarnessConfig()
        new_config, reasoning = propose_harness_changes(
            llm, "diagnosis text", config
        )

        # Verify clamping
        assert new_config.max_retries == 10  # clamped to upper bound
        assert new_config.min_length == 100  # clamped to lower bound
        assert new_config.max_similar == 1  # clamped to lower bound
        assert new_config.budget_seconds == 600  # clamped to upper bound
        assert new_config.planning_calls == 1  # clamped to lower bound
        # Invalid effort string should be ignored, keeping default
        assert new_config.planning_effort in {"low", "medium", "high"}


class TestOptimizeHarness:
    def test_saves_new_version_when_changed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HarnessConfigStore(Path(tmpdir) / "configs")
            store.save(HarnessConfig())  # v1 baseline

            llm = MagicMock()
            llm.return_value = json.dumps(
                {
                    "changes": {"max_retries": 5},
                    "reasoning": "More retries needed.",
                }
            )

            entries = [_make_entry(score=0.4)]
            version = optimize_harness(llm, entries, store)

            assert version >= 2
