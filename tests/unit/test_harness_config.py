"""Tests for HarnessConfig and HarnessConfigStore."""

import tempfile
from pathlib import Path

import pytest

from src.evolution.harness_config import HarnessConfig, HarnessConfigStore


class TestHarnessConfig:
    def test_defaults(self):
        cfg = HarnessConfig()
        # SelfVerification
        assert cfg.max_retries == 2
        assert cfg.min_length == 500
        assert cfg.verify_against_task is False
        assert cfg.required_sections == ["summary", "finding", "source"]
        # LoopDetection
        assert cfg.max_similar == 3
        assert cfg.max_total == 12
        assert cfg.max_file_edits == 5
        assert cfg.max_repeated_tools == 4
        # TimeBudget
        assert cfg.budget_seconds == 300
        assert cfg.warn_at == [0.6, 0.85]
        # ReasoningSandwich
        assert cfg.planning_effort == "high"
        assert cfg.implementation_effort == "medium"
        assert cfg.verification_effort == "high"
        assert cfg.planning_calls == 2
        # ContextAssembly
        assert cfg.detect_env is True

    def test_to_dict_and_from_dict(self):
        cfg = HarnessConfig(
            max_retries=5,
            min_length=1000,
            verify_against_task=True,
            required_sections=["abstract", "conclusion"],
            max_similar=10,
            max_total=50,
            max_file_edits=20,
            max_repeated_tools=8,
            budget_seconds=600,
            warn_at=[0.5, 0.9],
            planning_effort="low",
            implementation_effort="high",
            verification_effort="low",
            planning_calls=4,
            detect_env=False,
        )
        d = cfg.to_dict()
        restored = HarnessConfig.from_dict(d)
        assert restored.to_dict() == d

    def test_from_dict_filters_unknown_fields(self):
        d = {"max_retries": 7, "unknown_field": "should_be_ignored"}
        cfg = HarnessConfig.from_dict(d)
        assert cfg.max_retries == 7
        assert not hasattr(cfg, "unknown_field")


class TestHarnessConfigStore:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HarnessConfigStore(Path(tmp) / "configs")
            cfg = HarnessConfig(max_retries=5)
            ver = store.save(cfg, score=0.8)
            assert ver == 1
            loaded = store.load_best()
            assert loaded.max_retries == 5

    def test_load_best_returns_latest_version_ignoring_scores(self):
        """Ratchet semantics: the active config is the LATEST version, period.

        Even if an older version has a higher running score, the latest
        promoted version remains active. This prevents observed-score drift
        from silently rolling the active config back to an older one.
        """
        with tempfile.TemporaryDirectory() as tmp:
            store = HarnessConfigStore(Path(tmp) / "configs")
            store.save(HarnessConfig(max_retries=1), score=0.5)
            store.save(HarnessConfig(max_retries=2), score=0.9)
            store.save(HarnessConfig(max_retries=3), score=0.7)
            best = store.load_best()
            # v3 is latest → active, even though v2 has the highest score.
            assert best.max_retries == 3

    def test_load_best_returns_latest_when_unscored(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HarnessConfigStore(Path(tmp) / "configs")
            store.save(HarnessConfig(max_retries=1))
            store.save(HarnessConfig(max_retries=2))
            store.save(HarnessConfig(max_retries=3))
            best = store.load_best()
            assert best.max_retries == 3

    def test_promotion_score_is_frozen_under_observed_drift(self):
        """The frozen promotion_score must NOT change when update_score
        runs. Only the running ``score`` field drifts."""
        import json
        with tempfile.TemporaryDirectory() as tmp:
            store = HarnessConfigStore(Path(tmp) / "configs")
            store.save(HarnessConfig(), score=0.8)

            # Hammer it with bad observed scores
            for bad in [0.3, 0.2, 0.4, 0.25, 0.35]:
                store.update_score(1, bad)

            data = json.loads((Path(tmp) / "configs" / "v0001.json").read_text())
            assert data["promotion_score"] == 0.8, (
                "promotion_score must remain 0.8 even after observed drift"
            )
            # Running score should have moved (the optimizer's signal)
            assert data["score"] < 0.6
            # And promotion_score is the ONLY thing the active selector sees
            # (well, indirectly via the latest-version rule).

    def test_promotion_score_inherited_from_explicit_arg(self):
        """When save() is called with an explicit promotion_score that
        differs from the running score, the explicit value wins."""
        import json
        with tempfile.TemporaryDirectory() as tmp:
            store = HarnessConfigStore(Path(tmp) / "configs")
            # First version: baseline
            store.save(HarnessConfig(), score=0.8)
            # Second version: optimizer wants this version's frozen baseline
            # to inherit the parent's 0.8 even though current batch was 0.5
            store.save(HarnessConfig(max_retries=5), score=0.5, promotion_score=0.8)

            v2 = json.loads((Path(tmp) / "configs" / "v0002.json").read_text())
            assert v2["promotion_score"] == 0.8
            assert v2["score"] == 0.5

    def test_update_score_incremental(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HarnessConfigStore(Path(tmp) / "configs")
            store.save(HarnessConfig(), score=0.6)
            # Second score: avg = 0.6 + (0.8 - 0.6)/2 = 0.7
            store.update_score(1, 0.8)
            # Third score: avg = 0.7 + (0.5 - 0.7)/3 = 0.6333...
            store.update_score(1, 0.5)
            # Load the raw data to check
            import json
            with open(store._version_path(1)) as f:
                data = json.load(f)
            assert data["score_count"] == 3
            assert abs(data["score"] - 0.6333) < 0.01

    def test_get_latest_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HarnessConfigStore(Path(tmp) / "configs")
            assert store.get_latest_version() == 0
            store.save(HarnessConfig())
            assert store.get_latest_version() == 1
            store.save(HarnessConfig())
            assert store.get_latest_version() == 2


class TestRunLogTraceCapture:
    def test_run_log_entry_has_new_fields(self):
        from src.evolution.run_log import RunLogEntry

        entry = RunLogEntry(
            run_id="test-001",
            task="test",
            output="report",
            classification="partial",
            average_score=0.78,
            grader_results=[],
            prompt_version=7,
            harness_config_version=2,
            trace_path="/tmp/traces/test.json",
        )
        d = entry.to_dict()
        assert d["trace_path"] == "/tmp/traces/test.json"
        assert d["harness_config_version"] == 2

    def test_run_log_roundtrip_with_new_fields(self, tmp_path):
        from src.evolution.run_log import RunLog, RunLogEntry

        log = RunLog(tmp_path / "test.jsonl")
        log.append(
            RunLogEntry(
                run_id="r1",
                task="test",
                output="out",
                classification="partial",
                average_score=0.72,
                grader_results=[],
                prompt_version=7,
                harness_config_version=3,
                trace_path="/tmp/trace.json",
            )
        )
        entries = log.read_all()
        assert entries[0].harness_config_version == 3
        assert entries[0].trace_path == "/tmp/trace.json"


class TestToolConfig:
    def test_harness_config_has_tool_fields(self):
        from src.evolution.harness_config import HarnessConfig
        cfg = HarnessConfig()
        assert cfg.search_max_retries == 1
        assert cfg.search_retry_delay == 2
        assert cfg.search_max_results == 3
        assert cfg.search_depth == "basic"

    def test_tool_config_roundtrip(self):
        from src.evolution.harness_config import HarnessConfig
        cfg = HarnessConfig(search_max_retries=3, search_depth="advanced")
        d = cfg.to_dict()
        restored = HarnessConfig.from_dict(d)
        assert restored.search_max_retries == 3
        assert restored.search_depth == "advanced"


class TestCompletionChecks:
    def test_default_empty(self):
        from src.evolution.harness_config import HarnessConfig
        cfg = HarnessConfig()
        assert cfg.completion_checks == []

    def test_roundtrip(self):
        from src.evolution.harness_config import HarnessConfig
        checks = ["Has 3+ sources", "Addresses all sub-questions"]
        cfg = HarnessConfig(completion_checks=checks)
        d = cfg.to_dict()
        restored = HarnessConfig.from_dict(d)
        assert restored.completion_checks == checks


class TestCompletionCheckEvaluation:
    def test_evaluate_completion_checks_passes(self):
        from unittest.mock import MagicMock
        from evoagent.harness.middleware import _evaluate_completion_checks

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"passed": true}')

        result = _evaluate_completion_checks(
            mock_llm, "test task", "test output", ["has sources"]
        )
        assert result == []

    def test_evaluate_completion_checks_fails(self):
        from unittest.mock import MagicMock
        from evoagent.harness.middleware import _evaluate_completion_checks

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"passed": false, "failures": ["Missing sources"]}'
        )

        result = _evaluate_completion_checks(
            mock_llm, "test task", "test output", ["has 3+ sources"]
        )
        assert "Missing sources" in result


class TestHarnessConfigWiring:
    def test_build_middleware_from_config(self):
        from evoagent.harness.builder import default_middleware_stack

        cfg = HarnessConfig(budget_seconds=120, max_similar=7, max_retries=5, detect_env=True)
        stack = default_middleware_stack(harness_config=cfg)
        names = [type(m).__name__ for m in stack]
        assert "TimeBudgetMiddleware" in names
        assert "LoopDetectionMiddleware" in names
        assert "SelfVerificationMiddleware" in names
        assert "ContextAssemblyMiddleware" in names
        assert len(stack) == 6
