"""Tests for the parallel 4-grader analyzer pipeline."""

from typing import get_type_hints

import pytest

from src.evolution.analyzer import classify_trajectory
from src.evolution.state import AnalyzerState, GraderResult

# ---------------------------------------------------------------------------
# AnalyzerState type hints
# ---------------------------------------------------------------------------


class TestAnalyzerStateSchema:
    def test_has_claim_verification_field(self):
        hints = get_type_hints(AnalyzerState)
        assert "claim_verification" in hints, "AnalyzerState should have a claim_verification field"

    def test_claim_verification_type_is_grader_result(self):
        hints = get_type_hints(AnalyzerState)
        assert hints["claim_verification"] is GraderResult

    def test_all_expected_fields_present(self):
        hints = get_type_hints(AnalyzerState)
        expected = {
            "trajectory",
            "task_completion",
            "efficiency",
            "quality",
            "claim_verification",
            "classification",
            "average_score",
        }
        assert expected.issubset(set(hints.keys()))


# ---------------------------------------------------------------------------
# classify_trajectory with 4 graders
# ---------------------------------------------------------------------------


def _make_grader(name: str, score: float, passed: bool) -> GraderResult:
    return GraderResult(
        name=name,
        score=score,
        passed=passed,
        reasoning=f"{name} reasoning",
    )


class TestClassifyTrajectoryFourGraders:
    def test_all_pass_high_avg_is_successful(self):
        """4 graders all passing with high avg -> successful."""
        graders = [
            _make_grader("task_completion", 0.9, True),
            _make_grader("efficiency", 0.8, True),
            _make_grader("quality", 0.85, True),
            _make_grader("claim_verification", 0.9, True),
        ]
        classification, avg = classify_trajectory(graders)
        assert classification == "successful"
        assert avg >= 0.75

    def test_two_pass_medium_avg_is_partial(self):
        """2 graders pass, avg >= 0.6 but < 0.75 -> partial."""
        graders = [
            _make_grader("task_completion", 0.8, True),
            _make_grader("efficiency", 0.7, True),
            _make_grader("quality", 0.4, False),
            _make_grader("claim_verification", 0.5, False),
        ]
        classification, avg = classify_trajectory(graders)
        assert classification == "partial"
        # avg = (0.8 + 0.7 + 0.4 + 0.5) / 4 = 0.6
        assert avg >= 0.5

    def test_one_pass_low_avg_is_failed(self):
        """Only 1 grader passes, low avg -> failed."""
        graders = [
            _make_grader("task_completion", 0.8, True),
            _make_grader("efficiency", 0.3, False),
            _make_grader("quality", 0.2, False),
            _make_grader("claim_verification", 0.3, False),
        ]
        classification, avg = classify_trajectory(graders)
        assert classification == "failed"
        # avg = (0.8 + 0.3 + 0.2 + 0.3) / 4 = 0.4
        assert avg < 0.5


# ---------------------------------------------------------------------------
# Claim verification score included in average
# ---------------------------------------------------------------------------


class TestClaimVerificationInAverage:
    def test_claim_verification_affects_average(self):
        """Changing claim_verification score should change the average."""
        base = [
            _make_grader("task_completion", 0.8, True),
            _make_grader("efficiency", 0.8, True),
            _make_grader("quality", 0.8, True),
        ]

        # Without claim_verification (3 graders)
        _, avg_without = classify_trajectory(base)

        # With high claim_verification (4 graders)
        with_high = [*base, _make_grader("claim_verification", 1.0, True)]
        _, avg_with_high = classify_trajectory(with_high)

        # With low claim_verification (4 graders)
        with_low = [*base, _make_grader("claim_verification", 0.0, False)]
        _, avg_with_low = classify_trajectory(with_low)

        # The averages should differ
        assert avg_with_high > avg_with_low
        # 4th grader pulls average up or down
        assert avg_with_high >= avg_without
        assert avg_with_low < avg_without

    def test_four_grader_average_calculation(self):
        """Average of 4 graders is computed correctly."""
        graders = [
            _make_grader("task_completion", 0.8, True),
            _make_grader("efficiency", 0.6, True),
            _make_grader("quality", 0.7, True),
            _make_grader("claim_verification", 0.9, True),
        ]
        _, avg = classify_trajectory(graders)
        expected = (0.8 + 0.6 + 0.7 + 0.9) / 4
        assert avg == pytest.approx(expected, abs=0.001)
