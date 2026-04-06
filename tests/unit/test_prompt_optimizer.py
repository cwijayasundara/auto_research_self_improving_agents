"""Tests for prompt optimizer multi-candidate generation."""

from unittest.mock import MagicMock

from src.evolution.prompt_optimizer import generate_improved_prompt


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
