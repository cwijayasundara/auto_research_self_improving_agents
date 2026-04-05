"""Tests for multi-judge grading system (evoagent)."""

from unittest.mock import MagicMock

from evoagent.core.parsing import parse_llm_json
from evoagent.graders.multi_judge import MultiJudgeGrader


# ---------------------------------------------------------------------------
# parse_llm_json (replaces _parse_judge_response)
# ---------------------------------------------------------------------------


class TestParseLlmJson:
    def test_plain_json(self):
        text = '{"score": 0.85, "reasoning": "good"}'
        result = parse_llm_json(text)
        assert result["score"] == 0.85
        assert result["reasoning"] == "good"

    def test_markdown_code_block(self):
        text = '```json\n{"score": 0.7, "reasoning": "ok"}\n```'
        result = parse_llm_json(text)
        assert result["score"] == 0.7

    def test_invalid_json_returns_empty(self):
        result = parse_llm_json("not json at all")
        assert result == {}

    def test_empty_string(self):
        result = parse_llm_json("")
        assert result == {}


# ---------------------------------------------------------------------------
# MultiJudgeGrader (replaces multi_judge_task_completion / multi_judge_quality)
# ---------------------------------------------------------------------------


class TestMultiJudgeGrader:
    def test_returns_grader_result_structure(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(content='{"score": 0.9, "reasoning": "complete"}'),
            MagicMock(content='{"score": 0.8, "reasoning": "has evidence"}'),
            MagicMock(content='{"score": 0.85, "reasoning": "accurate"}'),
        ]
        grader = MultiJudgeGrader(llm, name="task_completion")
        result = grader.grade("research AI", "full report")
        assert result.name == "task_completion"
        assert isinstance(result.score, float)
        assert isinstance(result.passed, bool)
        assert isinstance(result.reasoning, str)

    def test_passing_score(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(content='{"score": 0.9, "reasoning": "a"}'),
            MagicMock(content='{"score": 0.85, "reasoning": "b"}'),
            MagicMock(content='{"score": 0.8, "reasoning": "c"}'),
        ]
        grader = MultiJudgeGrader(llm, name="task_completion")
        result = grader.grade("task", "output")
        assert result.score == 0.85
        assert result.passed is True

    def test_failing_score(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(content='{"score": 0.3, "reasoning": "a"}'),
            MagicMock(content='{"score": 0.4, "reasoning": "b"}'),
            MagicMock(content='{"score": 0.2, "reasoning": "c"}'),
        ]
        grader = MultiJudgeGrader(llm, name="task_completion")
        result = grader.grade("task", "output")
        assert result.score == 0.3
        assert result.passed is False

    def test_fallback_when_judges_fail(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            Exception("fail1"),
            Exception("fail2"),
            Exception("fail3"),
            MagicMock(content='{"score": 0.7, "reasoning": "fallback"}'),
        ]
        grader = MultiJudgeGrader(llm, name="task_completion")
        result = grader.grade("task", "output")
        assert result.score == 0.7

    def test_all_fail_returns_default(self):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("always fails")
        grader = MultiJudgeGrader(llm, name="task_completion")
        result = grader.grade("task", "output")
        assert result.score == 0.5
        assert "grader_error" in result.reasoning

    def test_low_agreement_flagged(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(content='{"score": 0.3, "reasoning": "low"}'),
            MagicMock(content='{"score": 0.9, "reasoning": "high"}'),
            MagicMock(content='{"score": 0.6, "reasoning": "mid"}'),
        ]
        grader = MultiJudgeGrader(llm, name="quality")
        result = grader.grade("task", "output")
        assert result.score == 0.6
        assert "low_agreement" in result.reasoning

    def test_quality_grader(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(content='{"score": 0.9, "reasoning": "structured"}'),
            MagicMock(content='{"score": 0.8, "reasoning": "deep"}'),
            MagicMock(content='{"score": 0.85, "reasoning": "relevant"}'),
        ]
        grader = MultiJudgeGrader(llm, name="quality")
        result = grader.grade("research AI", "full report")
        assert result.name == "quality"
        assert result.score == 0.85
        assert result.passed is True
