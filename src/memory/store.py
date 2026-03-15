"""Long-term memory store backed by JSON files.

Supports two memory namespaces:
- Episodic: Past run experiences (task, strategy, outcome)
- Semantic: Learned facts and patterns accumulated over time
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MemoryStore:
    """File-backed persistent memory store using JSON files."""

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.episodic_dir = memory_dir / "episodic"
        self.semantic_dir = memory_dir / "semantic"
        self.episodic_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_dir.mkdir(parents=True, exist_ok=True)

    def _namespace_dir(self, namespace: str) -> Path:
        if namespace == "episodic":
            return self.episodic_dir
        if namespace == "semantic":
            return self.semantic_dir
        msg = f"Unknown namespace: {namespace}"
        raise ValueError(msg)

    def store(self, namespace: str, key: str, data: dict[str, Any]) -> Path:
        """Store a memory document."""
        ns_dir = self._namespace_dir(namespace)
        data["_key"] = key
        data["_stored_at"] = datetime.now(UTC).isoformat()
        path = ns_dir / f"{key}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Stored %s memory: %s", namespace, key)
        return path

    def retrieve(self, namespace: str, key: str) -> dict[str, Any] | None:
        """Retrieve a specific memory by key."""
        ns_dir = self._namespace_dir(namespace)
        path = ns_dir / f"{key}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def list_all(self, namespace: str) -> list[dict[str, Any]]:
        """List all memories in a namespace."""
        ns_dir = self._namespace_dir(namespace)
        memories = []
        for path in sorted(ns_dir.glob("*.json")):
            with open(path) as f:
                memories.append(json.load(f))
        return memories

    def search(self, namespace: str, query: str) -> list[dict[str, Any]]:
        """Search memories by keyword matching."""
        query_terms = query.lower().split()
        results = []
        for memory in self.list_all(namespace):
            text = " ".join(
                str(v).lower() for v in memory.values() if isinstance(v, str)
            )
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                results.append({**memory, "_relevance_score": score})
        results.sort(key=lambda m: m.get("_relevance_score", 0), reverse=True)
        return results

    def count(self, namespace: str) -> int:
        """Count memories in a namespace."""
        ns_dir = self._namespace_dir(namespace)
        return len(list(ns_dir.glob("*.json")))

    def delete(self, namespace: str, key: str) -> bool:
        """Delete a specific memory."""
        ns_dir = self._namespace_dir(namespace)
        path = ns_dir / f"{key}.json"
        if path.exists():
            path.unlink()
            logger.info("Deleted %s memory: %s", namespace, key)
            return True
        return False
