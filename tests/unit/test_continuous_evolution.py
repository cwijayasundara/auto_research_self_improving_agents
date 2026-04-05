"""Integration tests for the continuous evolution pipeline."""

import json
from pathlib import Path

from src.evolution.harness_config import HarnessConfig, HarnessConfigStore
from src.evolution.run_log import RunLog, RunLogEntry
from src.agent.prompt_store import PromptStore


class TestContinuousEvolutionPipeline:
    def test_run_log_entry_roundtrip_with_harness_fields(self):
        """RunLogEntry serializes and deserializes with all new fields."""
        entry = RunLogEntry(
            run_id="test-001", task="test quantum", output="report...",
            classification="partial", average_score=0.78,
            grader_results=[
                {"name": "task_completion", "score": 0.4, "passed": False, "reasoning": "missed"},
                {"name": "efficiency", "score": 0.88, "passed": True, "reasoning": "ok"},
            ],
            prompt_version=7, harness_config_version=2, trace_path="/tmp/trace.json",
        )
        d = entry.to_dict()
        restored = RunLogEntry(**d)
        assert restored.harness_config_version == 2
        assert restored.trace_path == "/tmp/trace.json"
        assert restored.average_score == 0.78

    def test_harness_config_store_full_lifecycle(self, tmp_path):
        """Full lifecycle: save, score, load best."""
        store = HarnessConfigStore(tmp_path / "harness")

        # Save initial config
        store.save(HarnessConfig(), score=0.70)

        # Save evolved config with better score
        store.save(
            HarnessConfig(max_retries=3, budget_seconds=240),
            score=0.82,
        )

        # Best should be the evolved config
        best = store.load_best()
        assert best.max_retries == 3
        assert best.budget_seconds == 240

        # Incremental scoring pulls it down
        store.update_score(2, 0.60)
        # Now score = (0.82 + 0.60) / 2 = 0.71
        data = json.loads((tmp_path / "harness" / "v0002.json").read_text())
        assert abs(data["score"] - 0.71) < 0.01
        assert data["score_count"] == 2

    def test_prompt_and_harness_both_scored_from_runs(self, tmp_path):
        """Both prompt and harness config accumulate scores from run log entries."""
        prompt_store = PromptStore(tmp_path / "prompts")
        prompt_store.add_version("test prompt", score=None)

        harness_store = HarnessConfigStore(tmp_path / "harness")
        harness_store.save(HarnessConfig(), score=None)

        # Simulate 3 run scores
        for score in [0.75, 0.80, 0.70]:
            prompt_store.update_score(1, score)
            harness_store.update_score(1, score)

        # Both should have incremental average = 0.75
        p_data = json.loads((tmp_path / "prompts" / "v0001.json").read_text())
        h_data = json.loads((tmp_path / "harness" / "v0001.json").read_text())
        assert abs(p_data["score"] - 0.75) < 0.01
        assert abs(h_data["score"] - 0.75) < 0.01

    def test_run_log_with_harness_fields_persists(self, tmp_path):
        """Run log persists and reads harness-related fields correctly."""
        log = RunLog(tmp_path / "run_log.jsonl")

        log.append(RunLogEntry(
            run_id="r1", task="test", output="out",
            classification="partial", average_score=0.72,
            grader_results=[], prompt_version=7,
            harness_config_version=2, trace_path="/tmp/trace.json",
        ))
        log.append(RunLogEntry(
            run_id="r2", task="test2", output="out2",
            classification="successful", average_score=0.88,
            grader_results=[], prompt_version=7,
            harness_config_version=2,
        ))

        entries = log.read_all()
        assert len(entries) == 2
        assert entries[0].harness_config_version == 2
        assert entries[0].trace_path == "/tmp/trace.json"
        assert entries[1].trace_path == ""

        # Unprocessed count
        assert log.count_unprocessed() == 2

        # Mark one processed
        log.mark_processed({"r1"})
        assert log.count_unprocessed() == 1
