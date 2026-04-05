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

    def test_load_best_returns_highest_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HarnessConfigStore(Path(tmp) / "configs")
            store.save(HarnessConfig(max_retries=1), score=0.5)
            store.save(HarnessConfig(max_retries=2), score=0.9)
            store.save(HarnessConfig(max_retries=3), score=0.7)
            best = store.load_best()
            assert best.max_retries == 2

    def test_load_best_returns_latest_when_unscored(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = HarnessConfigStore(Path(tmp) / "configs")
            store.save(HarnessConfig(max_retries=1))
            store.save(HarnessConfig(max_retries=2))
            store.save(HarnessConfig(max_retries=3))
            best = store.load_best()
            assert best.max_retries == 3

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
