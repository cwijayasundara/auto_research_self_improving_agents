"""Tests for MultiJudgeGrader with mocked LLM."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from evoagent.graders.multi_judge import MultiJudgeGrader


def _mock_llm(*responses):
    llm = MagicMock()
    llm.invoke.side_effect = [AIMessage(content=r) for r in responses]
    return llm


def test_median_of_three_judges():
    llm = _mock_llm(
        '{"score": 0.9, "reasoning": "excellent"}',
        '{"score": 0.6, "reasoning": "ok"}',
        '{"score": 0.8, "reasoning": "good"}',
    )
    grader = MultiJudgeGrader(llm=llm, name="quality")
    result = grader.grade(task="test task", output="test output " * 100)
    assert result.score == 0.8
    assert result.name == "quality"


def test_fallback_on_judge_failure():
    llm = MagicMock()
    llm.invoke.side_effect = [
        Exception("timeout"),
        Exception("timeout"),
        Exception("timeout"),
        AIMessage(content='{"score": 0.7, "reasoning": "fallback"}'),
    ]
    grader = MultiJudgeGrader(llm=llm, name="test")
    result = grader.grade(task="test", output="output " * 100)
    assert result.score == 0.7


def test_flags_low_agreement():
    llm = _mock_llm(
        '{"score": 0.9, "reasoning": "great"}',
        '{"score": 0.3, "reasoning": "terrible"}',
        '{"score": 0.7, "reasoning": "ok"}',
    )
    grader = MultiJudgeGrader(llm=llm, name="test")
    result = grader.grade(task="test", output="output " * 100)
    assert "low_agreement" in result.reasoning
