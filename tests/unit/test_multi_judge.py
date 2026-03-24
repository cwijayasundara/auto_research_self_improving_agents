"""Tests for multi-judge grading system."""

from unittest.mock import MagicMock

from src.evolution.graders.multi_judge import (
    _aggregate_judge_scores,
    _parse_judge_response,
    _run_single_judge,
    multi_judge_quality,
    multi_judge_task_completion,
)

# ---------------------------------------------------------------------------
# _parse_judge_response
# ---------------------------------------------------------------------------


class TestParseJudgeResponse:
    def test_plain_json(self):
        text = '{"score": 0.85, "reasoning": "good"}'
        result = _parse_judge_response(text)
        assert result["score"] == 0.85
        assert result["reasoning"] == "good"

    def test_markdown_code_block(self):
        text = '```json\n{"score": 0.7, "reasoning": "ok"}\n```'
        result = _parse_judge_response(text)
        assert result["score"] == 0.7

    def test_invalid_json_returns_empty(self):
        result = _parse_judge_response("not json at all")
        assert result == {}

    def test_empty_string(self):
        result = _parse_judge_response("")
        assert result == {}


# ---------------------------------------------------------------------------
# _aggregate_judge_scores
# ---------------------------------------------------------------------------


class TestAggregateJudgeScores:
    def test_median_of_three(self):
        score, _reasoning = _aggregate_judge_scores([0.9, 0.7, 0.8], ["a", "b", "c"])
        assert score == 0.8

    def test_low_agreement_flag(self):
        score, reasoning = _aggregate_judge_scores([0.3, 0.9, 0.6], ["low", "high", "mid"])
        assert score == 0.6
        assert "low_agreement" in reasoning

    def test_high_agreement_no_flag(self):
        score, reasoning = _aggregate_judge_scores([0.8, 0.85, 0.82], ["a", "b", "c"])
        assert score == 0.82
        assert "low_agreement" not in reasoning

    def test_single_score(self):
        score, reasoning = _aggregate_judge_scores([0.75], ["only one"])
        assert score == 0.75
        assert reasoning == "only one"

    def test_empty_list(self):
        score, reasoning = _aggregate_judge_scores([], [])
        assert score == 0.5
        assert "grader_error" in reasoning


# ---------------------------------------------------------------------------
# _run_single_judge
# ---------------------------------------------------------------------------


class TestRunSingleJudge:
    def test_happy_path(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content='{"score": 0.85, "reasoning": "well done"}')
        score, reasoning = _run_single_judge(
            llm, "Judge: {task}\n{output}", "test task", "test output"
        )
        assert score == 0.85
        assert reasoning == "well done"

    def test_parse_error_no_score_key(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content='{"reasoning": "no score here"}')
        score, reasoning = _run_single_judge(
            llm, "Judge: {task}\n{output}", "test task", "test output"
        )
        assert score == 0.5
        assert "parse_error" in reasoning

    def test_exception_returns_none(self):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("LLM broke")
        score, reasoning = _run_single_judge(
            llm, "Judge: {task}\n{output}", "test task", "test output"
        )
        assert score is None
        assert "LLM broke" in reasoning

    def test_output_truncated_to_4000(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content='{"score": 0.8, "reasoning": "ok"}')
        long_output = "x" * 10000
        _run_single_judge(llm, "{task}\n{output}", "task", long_output)
        call_args = llm.invoke.call_args[0][0][0].content
        # The output portion should be truncated
        assert len(call_args) < 10000


# ---------------------------------------------------------------------------
# multi_judge_task_completion
# ---------------------------------------------------------------------------


class TestMultiJudgeTaskCompletion:
    def test_returns_grader_result_structure(self):
        llm = MagicMock()
        # 3 judges return different scores
        llm.invoke.side_effect = [
            MagicMock(content='{"score": 0.9, "reasoning": "complete"}'),
            MagicMock(content='{"score": 0.8, "reasoning": "has evidence"}'),
            MagicMock(content='{"score": 0.85, "reasoning": "accurate"}'),
        ]
        result = multi_judge_task_completion(llm, "research AI", "full report")
        assert result["name"] == "task_completion"
        assert isinstance(result["score"], float)
        assert isinstance(result["passed"], bool)
        assert isinstance(result["reasoning"], str)

    def test_passing_score(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(content='{"score": 0.9, "reasoning": "a"}'),
            MagicMock(content='{"score": 0.85, "reasoning": "b"}'),
            MagicMock(content='{"score": 0.8, "reasoning": "c"}'),
        ]
        result = multi_judge_task_completion(llm, "task", "output")
        assert result["score"] == 0.85
        assert result["passed"] is True

    def test_failing_score(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(content='{"score": 0.3, "reasoning": "a"}'),
            MagicMock(content='{"score": 0.4, "reasoning": "b"}'),
            MagicMock(content='{"score": 0.2, "reasoning": "c"}'),
        ]
        result = multi_judge_task_completion(llm, "task", "output")
        assert result["score"] == 0.3
        assert result["passed"] is False

    def test_fallback_when_judges_fail(self):
        llm = MagicMock()
        # First 3 calls raise exceptions, 4th (fallback) succeeds
        llm.invoke.side_effect = [
            Exception("fail1"),
            Exception("fail2"),
            Exception("fail3"),
            MagicMock(content='{"score": 0.7, "reasoning": "fallback"}'),
        ]
        result = multi_judge_task_completion(llm, "task", "output")
        assert result["score"] == 0.7

    def test_all_fail_returns_default(self):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("always fails")
        result = multi_judge_task_completion(llm, "task", "output")
        assert result["score"] == 0.5
        assert "grader_error" in result["reasoning"]


# ---------------------------------------------------------------------------
# multi_judge_quality
# ---------------------------------------------------------------------------


class TestMultiJudgeQuality:
    def test_returns_grader_result_structure(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(content='{"score": 0.9, "reasoning": "structured"}'),
            MagicMock(content='{"score": 0.8, "reasoning": "deep"}'),
            MagicMock(content='{"score": 0.85, "reasoning": "relevant"}'),
        ]
        result = multi_judge_quality(llm, "research AI", "full report")
        assert result["name"] == "quality"
        assert isinstance(result["score"], float)
        assert isinstance(result["passed"], bool)
        assert isinstance(result["reasoning"], str)

    def test_median_score(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(content='{"score": 0.7, "reasoning": "a"}'),
            MagicMock(content='{"score": 0.9, "reasoning": "b"}'),
            MagicMock(content='{"score": 0.8, "reasoning": "c"}'),
        ]
        result = multi_judge_quality(llm, "task", "output")
        assert result["score"] == 0.8
        assert result["passed"] is True

    def test_fallback_on_failure(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            Exception("fail1"),
            Exception("fail2"),
            Exception("fail3"),
            MagicMock(content='{"score": 0.6, "reasoning": "fallback quality"}'),
        ]
        result = multi_judge_quality(llm, "task", "output")
        assert result["score"] == 0.6
        assert result["passed"] is False
