"""Tests for prompt optimizer multi-candidate generation."""

from pathlib import Path
from unittest.mock import MagicMock

from src.agent.prompt_store import PromptStore
from src.evolution.prompt_optimizer import (
    generate_improved_prompt,
    optimize_prompt,
    validate_candidate_prompt,
)
from src.evolution.state import GraderResult


def _make_failure_info():
    return {
        "failure_analysis": "2 failures found.",
        "common_issues": ["Too short", "Missing sources"],
        "trace_digest": "",
        "dimension_breakdown": "",
    }


class TestMultiCandidatePrompt:
    def test_single_candidate_backward_compat(self):
        """n_candidates=1 works and calls LLM exactly once."""
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content="You are a research agent.\n{memory_context}"
        )
        result = generate_improved_prompt(
            llm,
            current_prompt="You are a researcher.",
            current_score=0.5,
            failure_info=_make_failure_info(),
            n_candidates=1,
        )
        assert "research agent" in result
        assert llm.invoke.call_count == 1

    def test_picks_least_drift_candidate(self):
        """With 2 candidates, picks the one closest in length to current."""
        current = "Short prompt."  # len ~13
        llm = MagicMock()
        # First candidate: very long
        long_prompt = "A" * 500 + "\n{memory_context}"
        # Second candidate: close to current length
        short_prompt = "Better prompt.\n{memory_context}"

        llm.invoke.side_effect = [
            MagicMock(content=long_prompt),
            MagicMock(content=short_prompt),
        ]
        result = generate_improved_prompt(
            llm,
            current_prompt=current,
            current_score=0.5,
            failure_info=_make_failure_info(),
            n_candidates=2,
        )
        assert result == short_prompt

    def test_returns_current_when_all_fail(self):
        """If all candidates fail autonomy completely, returns current prompt."""
        from unittest.mock import patch

        llm = MagicMock()
        # Every response contains an autonomy violation
        llm.invoke.return_value = MagicMock(
            content="Please ask the user which option they prefer."
        )
        current = "You are an autonomous agent."

        # Patch _strip_autonomy_violations to also fail so candidates are None
        with patch(
            "src.evolution.prompt_optimizer._strip_autonomy_violations",
            return_value="do you want something else?",
        ):
            result = generate_improved_prompt(
                llm,
                current_prompt=current,
                current_score=0.5,
                failure_info=_make_failure_info(),
                n_candidates=2,
            )
        assert result == current


class TestValidateCandidateDoesNotPersist:
    """Regression: validate_candidate_prompt must not write losing (or any)
    candidates to the prompt store. The candidate is run via prompt_override.
    """

    def _make_analysis(self, task: str, output: str):
        return {
            "task": task,
            "output": output,
            "classification": "partial",
            "average_score": 0.5,
            "grader_results": [],
        }

    def test_losing_candidate_not_written_to_store(self, tmp_path: Path):
        store = PromptStore(tmp_path / "prompts")
        store.add_version("baseline prompt v1\n{memory_context}", score=0.8)
        baseline_count = len(store.get_all_versions())
        baseline_latest = store.get_latest_version_number()

        analyses = [
            self._make_analysis("task A", "old output A"),
            self._make_analysis("task B", "old output B"),
        ]

        llm = MagicMock()
        # Pairwise judge always picks "A" (the first slot) — paired with the
        # random A/B swap inside pairwise_compare this is a noisy signal but
        # it's enough that the candidate cannot win 2-0 deterministically.
        llm.invoke.return_value = MagicMock(
            content='{"winner": "A", "confidence": "high", "reasoning": "x"}'
        )

        from unittest.mock import patch

        # Stub out the actual agent run so we don't spin up an LLM.
        with patch(
            "src.evolution.orchestrator._run_single_task",
            return_value={"output": "new output", "status": "completed"},
        ):
            validate_candidate_prompt(
                llm=llm,
                settings=MagicMock(),
                prompt_store=store,
                memory_store=MagicMock(),
                candidate_prompt="candidate prompt that should NEVER be saved",
                analyses=analyses,
            )

        # The store must be untouched regardless of validation outcome.
        assert len(store.get_all_versions()) == baseline_count
        assert store.get_latest_version_number() == baseline_latest
        for v in store.get_all_versions():
            assert "NEVER be saved" not in v.prompt

    def test_candidate_run_via_prompt_override(self, tmp_path: Path):
        """The candidate must reach _run_single_task via prompt_override,
        not via the prompt store."""
        store = PromptStore(tmp_path / "prompts")
        store.add_version("baseline\n{memory_context}", score=0.8)

        analyses = [
            self._make_analysis("task A", "old A"),
            self._make_analysis("task B", "old B"),
        ]

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='{"winner": "A", "confidence": "low", "reasoning": "x"}'
        )

        from unittest.mock import patch

        with patch(
            "src.evolution.orchestrator._run_single_task",
            return_value={"output": "new", "status": "completed"},
        ) as mock_run:
            validate_candidate_prompt(
                llm=llm,
                settings=MagicMock(),
                prompt_store=store,
                memory_store=MagicMock(),
                candidate_prompt="CANDIDATE-XYZ",
                analyses=analyses,
            )

        assert mock_run.call_count == 2
        for call in mock_run.call_args_list:
            assert call.kwargs.get("prompt_override") == "CANDIDATE-XYZ"


class TestOptimizePromptEndToEnd:
    """End-to-end tests for optimize_prompt covering the win and lose paths.

    These pin the bug fix at the level where the daemon actually consumes
    the result: optimize_prompt must return the *unchanged* version number
    when the candidate loses pairwise validation, so daemon.py:116's
    `if new_version != old_version` check does not fire.
    """

    def _make_analysis(self, task: str, output: str, score: float = 0.5):
        return {
            "task": task,
            "output": output,
            "classification": "partial",
            "average_score": score,
            "grader_results": [
                GraderResult(
                    name="task_completion",
                    score=score,
                    passed=False,
                    reasoning="missing detail",
                )
            ],
        }

    def _seed_store(self, tmp_path: Path) -> PromptStore:
        store = PromptStore(tmp_path / "prompts")
        store.add_version("baseline prompt\n{memory_context}", score=0.8)
        return store

    def test_losing_candidate_returns_unchanged_version(self, tmp_path: Path):
        """When validation rejects the candidate, optimize_prompt must
        return the same version number it started with — otherwise the
        daemon logs a misleading 'Prompt upgraded' message."""
        store = self._seed_store(tmp_path)
        starting_version = store.get_latest_version_number()
        starting_count = len(store.get_all_versions())

        analyses = [
            self._make_analysis("task A", "weak output A"),
            self._make_analysis("task B", "weak output B"),
        ]

        llm = MagicMock()
        # First two invokes: generate_improved_prompt's two candidates.
        # Subsequent invokes: pairwise judge — make it always pick "A"
        # so the candidate cannot win 2-0 deterministically.
        llm.invoke.side_effect = [
            MagicMock(content="candidate prompt 1\n{memory_context}"),
            MagicMock(content="candidate prompt 2\n{memory_context}"),
            MagicMock(
                content='{"winner": "A", "confidence": "high", "reasoning": "x"}'
            ),
            MagicMock(
                content='{"winner": "A", "confidence": "high", "reasoning": "x"}'
            ),
        ]

        from unittest.mock import patch

        # Pin random.random so pairwise_compare's A/B slot swap is
        # deterministic — without this, "A" maps to "new" ~50% of the
        # time and the candidate can win by chance, making the test flaky.
        with (
            patch(
                "src.evolution.orchestrator._run_single_task",
                return_value={"output": "candidate's new output", "status": "completed"},
            ),
            patch("src.evolution.prompt_optimizer.random.random", return_value=0.0),
        ):
            returned = optimize_prompt(
                llm=llm,
                prompt_store=store,
                analyses=analyses,
                settings=MagicMock(),
                memory_store=MagicMock(),
            )

        assert returned == starting_version, (
            "optimize_prompt must return the unchanged version on validation "
            "failure so daemon.py won't log a phantom upgrade"
        )
        assert len(store.get_all_versions()) == starting_count, (
            "no new prompt files should be written when validation fails"
        )

    def test_winning_candidate_persisted_with_lineage(self, tmp_path: Path):
        """When validation accepts the candidate, optimize_prompt must
        write a single new version with proper parent_version and
        feedback_summary set."""
        store = self._seed_store(tmp_path)
        starting_version = store.get_latest_version_number()

        analyses = [
            self._make_analysis("task A", "weak A"),
            self._make_analysis("task B", "weak B"),
        ]

        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(content="winning candidate prompt\n{memory_context}"),
            MagicMock(content="winning candidate prompt\n{memory_context}"),
            MagicMock(
                content='{"winner": "B", "confidence": "high", "reasoning": "x"}'
            ),
            MagicMock(
                content='{"winner": "B", "confidence": "high", "reasoning": "x"}'
            ),
        ]

        from unittest.mock import patch

        # Force pairwise_compare's random A/B swap to be deterministic so
        # "B" reliably maps to "new" — the candidate.
        with (
            patch(
                "src.evolution.orchestrator._run_single_task",
                return_value={"output": "strong new output", "status": "completed"},
            ),
            patch("src.evolution.prompt_optimizer.random.random", return_value=0.0),
        ):
            returned = optimize_prompt(
                llm=llm,
                prompt_store=store,
                analyses=analyses,
                settings=MagicMock(),
                memory_store=MagicMock(),
            )

        assert returned == starting_version + 1
        all_versions = store.get_all_versions()
        new_version = all_versions[-1]
        assert "winning candidate" in new_version.prompt
        assert new_version.parent_version == starting_version
        assert new_version.feedback_summary  # non-empty


class TestHoldoutValidation:
    """Pin the held-out validation path: when the daemon supplies a stable
    held-out task list, validate_candidate_prompt must use it instead of
    falling back to current-batch failures.

    Critical properties:
    1. Both champion AND candidate are run fresh on EACH held-out task
       (apples-to-apples comparison, no stale historical outputs)
    2. The held-out path is preferred when ≥2 tasks are supplied
    3. The fallback path runs (with a warning) only when no holdout is given
    """

    def _make_analysis(self, task: str, output: str):
        return {
            "task": task,
            "output": output,
            "classification": "partial",
            "average_score": 0.5,
            "grader_results": [
                GraderResult(name="task_completion", score=0.5, passed=False, reasoning="r"),
            ],
        }

    def test_holdout_path_runs_both_champion_and_candidate_fresh(self, tmp_path: Path):
        """For each held-out task, _run_single_task should be called TWICE:
        once with prompt_override=None (champion) and once with
        prompt_override=<candidate> (the candidate). 3 holdout tasks = 6 runs."""
        from src.evolution.prompt_optimizer import validate_candidate_prompt

        store = PromptStore(tmp_path / "prompts")
        store.add_version("champion prompt\n{memory_context}", score=0.8)

        holdout = ["task A", "task B", "task C"]

        llm = MagicMock()
        # Pairwise judge always picks "A" (the first slot). Combined with
        # the patched random.random=0 (no swap), A=old, so old wins.
        llm.invoke.return_value = MagicMock(
            content='{"winner": "A", "confidence": "high", "reasoning": "x"}'
        )

        from unittest.mock import patch
        with (
            patch(
                "src.evolution.orchestrator._run_single_task",
                return_value={"output": "fresh output", "status": "completed"},
            ) as mock_run,
            patch("src.evolution.prompt_optimizer.random.random", return_value=0.0),
        ):
            validate_candidate_prompt(
                llm=llm,
                settings=MagicMock(),
                prompt_store=store,
                memory_store=MagicMock(),
                candidate_prompt="CANDIDATE-XYZ",
                analyses=[],   # empty — held-out path should not touch this
                holdout_tasks=holdout,
            )

        # 3 holdout tasks * 2 runs each (champion + candidate) = 6 calls
        assert mock_run.call_count == 6

        # Half the calls should have prompt_override=None (champion fresh)
        # The other half should have prompt_override="CANDIDATE-XYZ"
        champion_calls = [
            c for c in mock_run.call_args_list
            if c.kwargs.get("prompt_override") is None
        ]
        candidate_calls = [
            c for c in mock_run.call_args_list
            if c.kwargs.get("prompt_override") == "CANDIDATE-XYZ"
        ]
        assert len(champion_calls) == 3
        assert len(candidate_calls) == 3

    def test_holdout_path_returns_true_when_candidate_wins(self, tmp_path: Path):
        from src.evolution.prompt_optimizer import validate_candidate_prompt

        store = PromptStore(tmp_path / "prompts")
        store.add_version("champion\n{memory_context}", score=0.8)

        holdout = ["t1", "t2", "t3"]
        llm = MagicMock()
        # Judge always picks "B"; with random.random=0 (no swap), B=new
        llm.invoke.return_value = MagicMock(
            content='{"winner": "B", "confidence": "high", "reasoning": "x"}'
        )

        from unittest.mock import patch
        with (
            patch(
                "src.evolution.orchestrator._run_single_task",
                return_value={"output": "fresh", "status": "completed"},
            ),
            patch("src.evolution.prompt_optimizer.random.random", return_value=0.0),
        ):
            result = validate_candidate_prompt(
                llm=llm,
                settings=MagicMock(),
                prompt_store=store,
                memory_store=MagicMock(),
                candidate_prompt="CANDIDATE",
                analyses=[],
                holdout_tasks=holdout,
            )

        assert result is True  # candidate won 3-0 on holdout

    def test_holdout_path_returns_false_when_champion_wins(self, tmp_path: Path):
        from src.evolution.prompt_optimizer import validate_candidate_prompt

        store = PromptStore(tmp_path / "prompts")
        store.add_version("champion\n{memory_context}", score=0.8)

        holdout = ["t1", "t2", "t3"]
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='{"winner": "A", "confidence": "high", "reasoning": "x"}'
        )

        from unittest.mock import patch
        with (
            patch(
                "src.evolution.orchestrator._run_single_task",
                return_value={"output": "fresh", "status": "completed"},
            ),
            patch("src.evolution.prompt_optimizer.random.random", return_value=0.0),
        ):
            result = validate_candidate_prompt(
                llm=llm,
                settings=MagicMock(),
                prompt_store=store,
                memory_store=MagicMock(),
                candidate_prompt="CANDIDATE",
                analyses=[],
                holdout_tasks=holdout,
            )

        assert result is False  # champion won 3-0 on holdout

    def test_holdout_path_treats_run_errors_as_loss_for_candidate(self, tmp_path: Path):
        """If a candidate run errors out, that task should count as a loss
        for the candidate (defensive — never promote a candidate that
        crashes)."""
        from src.evolution.prompt_optimizer import validate_candidate_prompt

        store = PromptStore(tmp_path / "prompts")
        store.add_version("champion\n{memory_context}", score=0.8)

        holdout = ["t1", "t2"]
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='{"winner": "B", "confidence": "high", "reasoning": "x"}'
        )

        # Champion always succeeds; candidate always errors
        call_count = [0]

        def fake_run(*args, **kwargs):
            call_count[0] += 1
            if kwargs.get("prompt_override") is not None:
                # Candidate run — return error
                return {"output": "", "status": "error"}
            return {"output": "champion output", "status": "completed"}

        from unittest.mock import patch
        with (
            patch("src.evolution.orchestrator._run_single_task", side_effect=fake_run),
            patch("src.evolution.prompt_optimizer.random.random", return_value=0.0),
        ):
            result = validate_candidate_prompt(
                llm=llm,
                settings=MagicMock(),
                prompt_store=store,
                memory_store=MagicMock(),
                candidate_prompt="CANDIDATE",
                analyses=[],
                holdout_tasks=holdout,
            )

        assert result is False  # candidate errored on every task → 0-2 loss

    def test_falls_back_to_batch_when_no_holdout(self, tmp_path: Path):
        """When holdout_tasks is None or too short, fall back to the
        legacy training-batch behavior (and log a warning)."""
        from src.evolution.prompt_optimizer import validate_candidate_prompt

        store = PromptStore(tmp_path / "prompts")
        store.add_version("champion\n{memory_context}", score=0.8)

        analyses = [
            self._make_analysis("task A", "old A"),
            self._make_analysis("task B", "old B"),
        ]

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='{"winner": "A", "confidence": "high", "reasoning": "x"}'
        )

        from unittest.mock import patch
        with (
            patch(
                "src.evolution.orchestrator._run_single_task",
                return_value={"output": "candidate output", "status": "completed"},
            ) as mock_run,
            patch("src.evolution.prompt_optimizer.random.random", return_value=0.0),
        ):
            validate_candidate_prompt(
                llm=llm,
                settings=MagicMock(),
                prompt_store=store,
                memory_store=MagicMock(),
                candidate_prompt="CANDIDATE",
                analyses=analyses,
                holdout_tasks=None,   # explicitly no holdout
            )

        # Fallback path: 2 tasks * 1 run (candidate only) = 2 calls
        assert mock_run.call_count == 2
        for c in mock_run.call_args_list:
            assert c.kwargs.get("prompt_override") == "CANDIDATE"

    def test_falls_back_when_holdout_too_short(self, tmp_path: Path):
        """A holdout list with <2 tasks is too small to gate on; fall back."""
        from src.evolution.prompt_optimizer import validate_candidate_prompt

        store = PromptStore(tmp_path / "prompts")
        store.add_version("champion\n{memory_context}", score=0.8)

        analyses = [
            self._make_analysis("task A", "old A"),
            self._make_analysis("task B", "old B"),
        ]

        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='{"winner": "A", "confidence": "high", "reasoning": "x"}'
        )

        from unittest.mock import patch
        with (
            patch(
                "src.evolution.orchestrator._run_single_task",
                return_value={"output": "candidate output", "status": "completed"},
            ) as mock_run,
            patch("src.evolution.prompt_optimizer.random.random", return_value=0.0),
        ):
            validate_candidate_prompt(
                llm=llm,
                settings=MagicMock(),
                prompt_store=store,
                memory_store=MagicMock(),
                candidate_prompt="CANDIDATE",
                analyses=analyses,
                holdout_tasks=["only one task"],   # too short
            )

        # Should fall back: 2 batch tasks * 1 run = 2
        assert mock_run.call_count == 2
