"""Tests for memory compression."""

from evoagent.memory.compression import compress_context, deduplicate_semantic
from evoagent.memory.store import FileMemoryStore


def test_compress_empty_memory(tmp_path):
    store = FileMemoryStore(tmp_path)
    context = compress_context(store, task="test", token_budget=1000)
    assert isinstance(context, str)


def test_compress_respects_budget(tmp_path):
    store = FileMemoryStore(tmp_path)
    for i in range(20):
        store.store("semantic", f"fact-{i}", {"content": f"Fact number {i} " * 50})
    context = compress_context(store, task="test", token_budget=500)
    assert len(context) < 3000


def test_compress_includes_meta_instructions(tmp_path):
    store = FileMemoryStore(tmp_path)
    store.store("semantic", "meta-1", {"type": "meta_instruction", "content": "Always verify"})
    store.store("semantic", "fact-1", {"content": "Regular fact"})
    context = compress_context(store, task="test", token_budget=2000)
    assert "Always verify" in context


def test_deduplicate_semantic(tmp_path):
    store = FileMemoryStore(tmp_path)
    store.store("semantic", "a", {"content": "quantum computing is fast"})
    store.store("semantic", "b", {"content": "quantum computing is very fast"})
    store.store("semantic", "c", {"content": "weather is nice today"})
    removed = deduplicate_semantic(store, threshold=0.6)
    assert removed >= 1
    assert store.count("semantic") <= 2
