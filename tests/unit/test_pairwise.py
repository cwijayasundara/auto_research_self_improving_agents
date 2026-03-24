"""Tests for pairwise comparison in prompt optimization."""

import json
from unittest.mock import MagicMock, patch

from src.evolution.prompt_optimizer import (
    _select_tasks_for_mini_scoring,
    pairwise_compare,
)

# ---------------------------------------------------------------------------
# _select_tasks_for_mini_scoring
# ---------------------------------------------------------------------------


def _make_analysis(task, classification, score):
    return {
        "run_id": "test",
        "task": task,
        "classification": classification,
        "average_score": score,
        "grader_results": [],
        "output": f"output for {task}",
        "tool_calls": [],
    }


class TestSelectTasksForMiniScoring:
    def test_empty_analyses(self):
        assert _select_tasks_for_mini_scoring([]) == []

    def test_prioritizes_failures(self):
        analyses = [
            _make_analysis("good task", "successful", 0.9),
            _make_analysis("bad task", "failed", 0.2),
            _make_analysis("ok task", "partial", 0.5),
        ]
        selected = _select_tasks_for_mini_scoring(analyses, max_tasks=2)
        tasks = [s["task"] for s in selected]
        assert "bad task" in tasks
        assert "ok task" in tasks
        assert "good task" not in tasks

    def test_fills_from_successful_when_few_failures(self):
        analyses = [
            _make_analysis("good task", "successful", 0.9),
            _make_analysis("bad task", "failed", 0.2),
        ]
        selected = _select_tasks_for_mini_scoring(analyses, max_tasks=2)
        tasks = [s["task"] for s in selected]
        assert "bad task" in tasks
        assert "good task" in tasks

    def test_sorts_failures_by_score(self):
        analyses = [
            _make_analysis("mid fail", "failed", 0.4),
            _make_analysis("worst fail", "failed", 0.1),
            _make_analysis("ok fail", "partial", 0.6),
        ]
        selected = _select_tasks_for_mini_scoring(analyses, max_tasks=2)
        # Should pick worst two by score
        assert selected[0]["task"] == "worst fail"
        assert selected[1]["task"] == "mid fail"

    def test_max_tasks_respected(self):
        analyses = [_make_analysis(f"task {i}", "failed", 0.1 * i) for i in range(5)]
        selected = _select_tasks_for_mini_scoring(analyses, max_tasks=2)
        assert len(selected) == 2


# ---------------------------------------------------------------------------
# pairwise_compare
# ---------------------------------------------------------------------------


class TestPairwiseCompare:
    def _mock_llm(self, response_json):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content=json.dumps(response_json))
        return llm

    @patch("src.evolution.prompt_optimizer.random")
    def test_winner_a_no_swap(self, mock_random):
        """When random < 0.5 (no swap), A=output_a, B=output_b. LLM says A wins -> label_a wins."""
        mock_random.random.return_value = 0.3  # no swap (< 0.5)
        llm = self._mock_llm(
            {
                "winner": "A",
                "confidence": "high",
                "reasoning": "A is better",
            }
        )
        result = pairwise_compare(llm, "test task", "out_a", "out_b", "old", "new")
        assert result["winner"] == "old"
        assert result["confidence"] == "high"

    @patch("src.evolution.prompt_optimizer.random")
    def test_winner_b_no_swap(self, mock_random):
        """When random < 0.5 (no swap), LLM says B wins -> label_b wins."""
        mock_random.random.return_value = 0.3  # no swap (< 0.5)
        llm = self._mock_llm(
            {
                "winner": "B",
                "confidence": "medium",
                "reasoning": "B is better",
            }
        )
        result = pairwise_compare(llm, "test task", "out_a", "out_b", "old", "new")
        assert result["winner"] == "new"

    @patch("src.evolution.prompt_optimizer.random")
    def test_winner_a_with_swap(self, mock_random):
        """When random >= 0.5 (swap), A=output_b, B=output_a. LLM says A wins -> label_b wins."""
        mock_random.random.return_value = 0.7  # swap (>= 0.5)
        llm = self._mock_llm(
            {
                "winner": "A",
                "confidence": "high",
                "reasoning": "A is better",
            }
        )
        result = pairwise_compare(llm, "test task", "out_a", "out_b", "old", "new")
        # Swapped: position A = output_b -> label_b = "new"
        assert result["winner"] == "new"

    @patch("src.evolution.prompt_optimizer.random")
    def test_winner_b_with_swap(self, mock_random):
        """When random >= 0.5 (swap), LLM says B wins -> label_a wins."""
        mock_random.random.return_value = 0.7  # swap (>= 0.5)
        llm = self._mock_llm(
            {
                "winner": "B",
                "confidence": "low",
                "reasoning": "B is better",
            }
        )
        result = pairwise_compare(llm, "test task", "out_a", "out_b", "old", "new")
        # Swapped: position B = output_a -> label_a = "old"
        assert result["winner"] == "old"

    @patch("src.evolution.prompt_optimizer.random")
    def test_handles_llm_error(self, mock_random):
        """On LLM error, returns label_a as winner with low confidence."""
        mock_random.random.return_value = 0.3
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("API error")
        result = pairwise_compare(llm, "test task", "out_a", "out_b", "old", "new")
        assert result["winner"] == "old"
        assert result["confidence"] == "low"

    @patch("src.evolution.prompt_optimizer.random")
    def test_handles_markdown_wrapped_json(self, mock_random):
        """LLM sometimes wraps JSON in markdown code blocks."""
        mock_random.random.return_value = 0.3  # no swap
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='```json\n{"winner": "B", "confidence": "high", "reasoning": "better"}\n```'
        )
        result = pairwise_compare(llm, "test task", "out_a", "out_b", "old", "new")
        assert result["winner"] == "new"
