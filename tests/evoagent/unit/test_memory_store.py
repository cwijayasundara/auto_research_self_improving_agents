"""Tests for FileMemoryStore."""

from evoagent.memory.store import FileMemoryStore


def test_store_and_retrieve(tmp_path):
    store = FileMemoryStore(tmp_path)
    store.store("episodic", "run-1", {"task": "test", "score": 0.8})
    result = store.retrieve("episodic", "run-1")
    assert result is not None
    assert result["task"] == "test"
    assert result["score"] == 0.8


def test_retrieve_missing_key(tmp_path):
    store = FileMemoryStore(tmp_path)
    assert store.retrieve("episodic", "nonexistent") is None


def test_list_all(tmp_path):
    store = FileMemoryStore(tmp_path)
    store.store("semantic", "fact-1", {"content": "Python is great"})
    store.store("semantic", "fact-2", {"content": "LangChain is useful"})
    all_items = store.list_all("semantic")
    assert len(all_items) == 2
    assert "fact-1" in all_items
    assert "fact-2" in all_items


def test_search(tmp_path):
    store = FileMemoryStore(tmp_path)
    store.store("semantic", "fact-1", {"content": "quantum computing advances"})
    store.store("semantic", "fact-2", {"content": "weather patterns today"})
    results = store.search("semantic", "quantum")
    assert len(results) >= 1
    assert any("quantum" in str(r) for r in results)


def test_count(tmp_path):
    store = FileMemoryStore(tmp_path)
    assert store.count("episodic") == 0
    store.store("episodic", "run-1", {"x": 1})
    assert store.count("episodic") == 1


def test_delete(tmp_path):
    store = FileMemoryStore(tmp_path)
    store.store("episodic", "run-1", {"x": 1})
    assert store.delete("episodic", "run-1") is True
    assert store.retrieve("episodic", "run-1") is None
    assert store.delete("episodic", "run-1") is False


def test_unknown_namespace_raises(tmp_path):
    store = FileMemoryStore(tmp_path)
    try:
        store.store("unknown_ns", "key", {})
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass
