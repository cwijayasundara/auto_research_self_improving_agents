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
        llm.invoke.return_value = MagicMock(
            content=json.dumps(
                {
                    "changes": {
                        "max_retries": 4,
                        "min_length": 1000,
                        "planning_effort": "high",
                    },
                    "reasoning": "Increasing retries to improve reliability.",
                }
            )
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
        llm.invoke.return_value = MagicMock(
            content=json.dumps(
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


class TestMultiCandidate:
    def test_propose_picks_most_conservative(self):
        """With 2 candidates, picks the one with fewer changes."""
        llm = MagicMock()
        # First call: changes 3 params, second call: changes 1 param
        llm.invoke.side_effect = [
            MagicMock(
                content=json.dumps(
                    {
                        "changes": {
                            "max_retries": 5,
                            "budget_seconds": 120,
                            "max_similar": 8,
                        },
                        "reasoning": "aggressive",
                    }
                )
            ),
            MagicMock(
                content=json.dumps(
                    {
                        "changes": {"max_retries": 3},
                        "reasoning": "conservative",
                    }
                )
            ),
        ]
        cfg = HarnessConfig()
        new_cfg, reasoning = propose_harness_changes(
            llm, "diagnosis", cfg, n_candidates=2
        )
        # Should pick the conservative one (1 change vs 3)
        assert new_cfg.max_retries == 3
        assert new_cfg.budget_seconds == 300  # unchanged (default)
        assert reasoning == "conservative"

    def test_propose_single_candidate_backward_compat(self):
        """n_candidates=1 still works (backward compatibility)."""
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content=json.dumps(
                {
                    "changes": {"max_retries": 5},
                    "reasoning": "more retries",
                }
            )
        )
        cfg = HarnessConfig()
        new_cfg, reasoning = propose_harness_changes(
            llm, "diagnosis", cfg, n_candidates=1
        )
        assert new_cfg.max_retries == 5
        assert llm.invoke.call_count == 1

    def test_propose_falls_back_on_all_failures(self):
        """If all candidates fail to parse, returns current config."""
        llm = MagicMock()
        llm.invoke.side_effect = Exception("LLM error")
        cfg = HarnessConfig()
        new_cfg, reasoning = propose_harness_changes(
            llm, "diagnosis", cfg, n_candidates=2
        )
        # Should return current config unchanged
        assert new_cfg.max_retries == cfg.max_retries
        assert new_cfg.budget_seconds == cfg.budget_seconds


class TestOptimizeHarness:
    def test_saves_new_version_when_changed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HarnessConfigStore(Path(tmpdir) / "configs")
            store.save(HarnessConfig())  # v1 baseline

            llm = MagicMock()
            llm.invoke.return_value = MagicMock(
                content=json.dumps(
                    {
                        "changes": {"max_retries": 5},
                        "reasoning": "More retries needed.",
                    }
                )
            )

            entries = [_make_entry(score=0.4)]
            version = optimize_harness(llm, entries, store)

            assert version >= 2
