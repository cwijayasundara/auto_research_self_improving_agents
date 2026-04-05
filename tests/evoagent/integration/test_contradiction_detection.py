"""Tests for memory contradiction detection."""

from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from evoagent.evolution.sleep_review import detect_contradictions
from evoagent.memory.store import FileMemoryStore


def test_no_contradictions_in_diverse_memories(tmp_path):
    store = FileMemoryStore(tmp_path)
    store.store("semantic", "a", {"content": "quantum computing uses qubits"})
    store.store("semantic", "b", {"content": "weather patterns affect agriculture"})
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content='{"contradictions": []}')
    removed = detect_contradictions(llm, store)
    assert removed == 0


def test_detects_and_removes_contradiction(tmp_path):
    store = FileMemoryStore(tmp_path)
    store.store("semantic", "old", {"content": "Tavily always returns structured JSON results"})
    store.store("semantic", "new", {"content": "Tavily often returns error dicts instead of results"})
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content='{"contradictions": [{"fact_a": "Tavily always returns structured JSON results", '
        '"fact_b": "Tavily often returns error dicts instead of results", '
        '"resolution": "Tavily sometimes returns errors", "keep": "new"}]}'
    )
    removed = detect_contradictions(llm, store)
    assert removed >= 1
    assert store.retrieve("semantic", "new") is not None
    assert store.retrieve("semantic", "old") is None


def test_empty_memory_returns_zero(tmp_path):
    store = FileMemoryStore(tmp_path)
    llm = MagicMock()
    removed = detect_contradictions(llm, store)
    assert removed == 0


def test_llm_error_returns_zero(tmp_path):
    store = FileMemoryStore(tmp_path)
    store.store("semantic", "a", {"content": "fact A about topic X"})
    store.store("semantic", "b", {"content": "fact A about topic X is wrong"})
    llm = MagicMock()
    llm.invoke.side_effect = Exception("LLM broke")
    removed = detect_contradictions(llm, store)
    assert removed == 0


def test_skips_meta_instructions(tmp_path):
    store = FileMemoryStore(tmp_path)
    store.store("semantic", "meta-1", {"type": "meta_instruction", "content": "always verify"})
    store.store("semantic", "meta-2", {"type": "meta_instruction", "content": "always verify facts"})
    llm = MagicMock()
    # Should not even be called since meta_instructions are skipped
    removed = detect_contradictions(llm, store)
    assert removed == 0
