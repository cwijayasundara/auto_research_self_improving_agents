"""Tests for the harness promotion gate (validate_candidate_harness).

Mirror of the prompt held-out validation tests. Pins:

1. Both current AND candidate harness configs are run fresh on EACH
   held-out task (apples-to-apples comparison)
2. Gate accepts when candidate wins strict majority
3. Gate rejects when champion wins
4. Gate is defensive: candidate run errors count as losses
5. optimize_harness wires the gate when holdout + dependencies are supplied
6. optimize_harness falls back to legacy "blind promote" with a warning
   when those are not supplied
7. Promotion gate failure → no new version saved on disk

The point of all this: the recent harness v2 (the "loosen min_length"
hallucinated remedy) would have been REJECTED by this gate because the
proposed change would not have measurably improved held-out outputs.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.evolution.harness_config import HarnessConfig, HarnessConfigStore
from src.evolution.harness_optimizer import (
    optimize_harness,
    validate_candidate_harness,
)


class TestValidateCandidateHarness:
    def test_runs_both_configs_fresh_per_task(self, tmp_path: Path):
        """3 held-out tasks → 6 _run_single_task calls (3 with current
        config, 3 with candidate)."""
        current = HarnessConfig(min_length=500)
        candidate = HarnessConfig(min_length=300)
        holdout = ["task A", "task B", "task C"]

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='{"winner": "A", "confidence": "high", "reasoning": "x"}'
        )

        with (
            patch(
                "src.evolution.orchestrator._run_single_task",
                return_value={"output": "fresh", "status": "completed"},
            ) as mock_run,
            patch("src.evolution.prompt_optimizer.random.random", return_value=0.0),
        ):
            validate_candidate_harness(
                llm=llm,
                settings=MagicMock(),
                prompt_store=MagicMock(),
                memory_store=MagicMock(),
                current_config=current,
                candidate_config=candidate,
                holdout_tasks=holdout,
            )

        assert mock_run.call_count == 6
        old_runs = [c for c in mock_run.call_args_list if c.kwargs.get("harness_override") is current]
        new_runs = [c for c in mock_run.call_args_list if c.kwargs.get("harness_override") is candidate]
        assert len(old_runs) == 3
        assert len(new_runs) == 3

    def test_returns_true_when_candidate_wins(self):
        current = HarnessConfig(min_length=500)
        candidate = HarnessConfig(min_length=300)
        holdout = ["t1", "t2", "t3"]

        llm = MagicMock()
        # Judge always picks "B"; with random.random=0 (no swap), B=new
        llm.invoke.return_value = MagicMock(
            content='{"winner": "B", "confidence": "high", "reasoning": "x"}'
        )

        with (
            patch(
                "src.evolution.orchestrator._run_single_task",
                return_value={"output": "fresh", "status": "completed"},
            ),
            patch("src.evolution.prompt_optimizer.random.random", return_value=0.0),
        ):
            result = validate_candidate_harness(
                llm=llm,
                settings=MagicMock(),
                prompt_store=MagicMock(),
                memory_store=MagicMock(),
                current_config=current,
                candidate_config=candidate,
                holdout_tasks=holdout,
            )

        assert result is True  # candidate won 3-0

    def test_returns_false_when_champion_wins(self):
        current = HarnessConfig(min_length=500)
        candidate = HarnessConfig(min_length=300)
        holdout = ["t1", "t2", "t3"]

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='{"winner": "A", "confidence": "high", "reasoning": "x"}'
        )

        with (
            patch(
                "src.evolution.orchestrator._run_single_task",
                return_value={"output": "fresh", "status": "completed"},
            ),
            patch("src.evolution.prompt_optimizer.random.random", return_value=0.0),
        ):
            result = validate_candidate_harness(
                llm=llm,
                settings=MagicMock(),
                prompt_store=MagicMock(),
                memory_store=MagicMock(),
                current_config=current,
                candidate_config=candidate,
                holdout_tasks=holdout,
            )

        assert result is False

    def test_candidate_errors_count_as_losses(self):
        """Defensive: never promote a config that crashes the agent."""
        current = HarnessConfig(min_length=500)
        candidate = HarnessConfig(min_length=300)
        holdout = ["t1", "t2"]

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='{"winner": "B", "confidence": "high", "reasoning": "x"}'
        )

        def fake_run(*args, **kwargs):
            if kwargs.get("harness_override") is candidate:
                return {"output": "", "status": "error"}
            return {"output": "champion fine", "status": "completed"}

        with (
            patch("src.evolution.orchestrator._run_single_task", side_effect=fake_run),
            patch("src.evolution.prompt_optimizer.random.random", return_value=0.0),
        ):
            result = validate_candidate_harness(
                llm=llm,
                settings=MagicMock(),
                prompt_store=MagicMock(),
                memory_store=MagicMock(),
                current_config=current,
                candidate_config=candidate,
                holdout_tasks=holdout,
            )

        assert result is False

    def test_skips_gate_when_holdout_too_small(self):
        """<2 holdout tasks → skip the gate and accept (consistent with
        the prompt validator's "not enough tasks → accept" branch)."""
        current = HarnessConfig()
        candidate = HarnessConfig(min_length=300)

        with patch("src.evolution.orchestrator._run_single_task") as mock_run:
            result = validate_candidate_harness(
                llm=MagicMock(),
                settings=MagicMock(),
                prompt_store=MagicMock(),
                memory_store=MagicMock(),
                current_config=current,
                candidate_config=candidate,
                holdout_tasks=["only one"],
            )

        assert result is True
        # Importantly: no runs were attempted
        assert mock_run.call_count == 0


class TestOptimizeHarnessGate:
    """End-to-end: optimize_harness must call validate_candidate_harness
    when holdout + dependencies are supplied, and must NOT save a new
    version when validation fails."""

    def _entry(self, score: float = 0.5):
        # Lightweight stand-in for RunLogEntry; only fields the optimizer reads
        entry = MagicMock()
        entry.average_score = score
        entry.task = "task"
        entry.output = "output"
        entry.classification = "partial"
        entry.trace_path = ""
        entry.run_id = "x"
        return entry

    def test_optimize_harness_skips_save_when_validation_fails(self, tmp_path: Path):
        store = HarnessConfigStore(tmp_path / "harness")
        store.save(HarnessConfig(min_length=500), score=0.7)
        starting_version = store.get_latest_version()

        # Force the proposer to return a different config so we get past
        # the "no changes" early-return
        new_cfg = HarnessConfig(min_length=300)

        llm = MagicMock()
        # Pairwise judge always picks "A"; with random.random=0 (no swap),
        # A=old, so the candidate loses 0-3
        llm.invoke.return_value = MagicMock(
            content='{"winner": "A", "confidence": "high", "reasoning": "x"}'
        )

        with (
            patch(
                "src.evolution.harness_optimizer.propose_harness_changes",
                return_value=(new_cfg, "loosen min_length"),
            ),
            patch(
                "src.evolution.orchestrator._run_single_task",
                return_value={"output": "fresh", "status": "completed"},
            ),
            patch("src.evolution.prompt_optimizer.random.random", return_value=0.0),
        ):
            returned = optimize_harness(
                llm=llm,
                entries=[self._entry(0.5), self._entry(0.4)],
                config_store=store,
                settings=MagicMock(),
                prompt_store=MagicMock(),
                memory_store=MagicMock(),
                holdout_tasks=["t1", "t2", "t3"],
            )

        assert returned == starting_version, (
            "optimize_harness must return the unchanged version when the "
            "pairwise gate rejects the candidate"
        )
        # No new file written
        assert store.get_latest_version() == starting_version

    def test_optimize_harness_promotes_when_validation_passes(self, tmp_path: Path):
        store = HarnessConfigStore(tmp_path / "harness")
        store.save(HarnessConfig(min_length=500), score=0.7)
        starting_version = store.get_latest_version()

        new_cfg = HarnessConfig(min_length=300)

        llm = MagicMock()
        # Judge picks "B" → with no swap, B=new → candidate wins 3-0
        llm.invoke.return_value = MagicMock(
            content='{"winner": "B", "confidence": "high", "reasoning": "x"}'
        )

        with (
            patch(
                "src.evolution.harness_optimizer.propose_harness_changes",
                return_value=(new_cfg, "loosen min_length"),
            ),
            patch(
                "src.evolution.orchestrator._run_single_task",
                return_value={"output": "fresh", "status": "completed"},
            ),
            patch("src.evolution.prompt_optimizer.random.random", return_value=0.0),
        ):
            returned = optimize_harness(
                llm=llm,
                entries=[self._entry(0.5), self._entry(0.4)],
                config_store=store,
                settings=MagicMock(),
                prompt_store=MagicMock(),
                memory_store=MagicMock(),
                holdout_tasks=["t1", "t2", "t3"],
            )

        assert returned == starting_version + 1
        assert store.get_latest_version() == starting_version + 1

    def test_optimize_harness_legacy_path_still_works_without_holdout(self, tmp_path: Path):
        """When holdout/deps are not supplied, the legacy "blind promote"
        path runs (with a warning). This preserves backward compat for any
        callers that haven't been updated to pass the gate dependencies."""
        store = HarnessConfigStore(tmp_path / "harness")
        store.save(HarnessConfig(min_length=500), score=0.7)
        starting_version = store.get_latest_version()

        new_cfg = HarnessConfig(min_length=300)

        with patch(
            "src.evolution.harness_optimizer.propose_harness_changes",
            return_value=(new_cfg, "loosen min_length"),
        ):
            returned = optimize_harness(
                llm=MagicMock(),
                entries=[self._entry(0.5)],
                config_store=store,
                # No settings/prompt_store/memory_store/holdout_tasks
            )

        assert returned == starting_version + 1
