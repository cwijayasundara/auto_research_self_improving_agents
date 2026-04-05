"""Tests for pre-completion task verification."""

from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from evoagent.harness.middleware import verify_output_against_task


def test_verify_addressed_returns_none():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content='{"addressed": true, "missing": []}')
    issues = verify_output_against_task(llm, "Research quantum computing", "# Report\n..." + "x" * 500)
    assert issues is None


def test_verify_not_addressed_returns_missing():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content='{"addressed": false, "missing": ["No discussion of error correction"]}'
    )
    issues = verify_output_against_task(llm, "Research quantum error correction", "# Report\n..." + "x" * 500)
    assert issues is not None
    assert len(issues) >= 1


def test_verify_llm_error_returns_none():
    llm = MagicMock()
    llm.invoke.side_effect = Exception("LLM broke")
    issues = verify_output_against_task(llm, "task", "output " * 200)
    assert issues is None  # fail-open
