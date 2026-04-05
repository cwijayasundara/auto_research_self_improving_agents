"""File-backed memory store implementing MemoryBackend protocol."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evoagent.core.protocols import MemoryBackend

logger = logging.getLogger(__name__)


class FileMemoryStore(MemoryBackend):
    """JSON file-backed persistent memory store."""

    NAMESPACES = frozenset({"episodic", "semantic"})

    def __init__(self, memory_dir: Path | str) -> None:
        self.memory_dir = Path(memory_dir)
        for ns in self.NAMESPACES:
            (self.memory_dir / ns).mkdir(parents=True, exist_ok=True)

    def _ns_dir(self, namespace: str) -> Path:
        if namespace not in self.NAMESPACES:
            msg = f"Unknown namespace: {namespace}. Must be one of {self.NAMESPACES}"
            raise ValueError(msg)
        return self.memory_dir / namespace

    def store(self, namespace: str, key: str, data: dict[str, Any]) -> None:
        ns_dir = self._ns_dir(namespace)
        data["_key"] = key
        data["_stored_at"] = datetime.now(UTC).isoformat()
        path = ns_dir / f"{key}.json"
        path.write_text(json.dumps(data, indent=2))

    def retrieve(self, namespace: str, key: str) -> dict[str, Any] | None:
        path = self._ns_dir(namespace) / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def list_all(self, namespace: str) -> dict[str, dict[str, Any]]:
        ns_dir = self._ns_dir(namespace)
        result: dict[str, dict[str, Any]] = {}
        for path in sorted(ns_dir.glob("*.json")):
            data = json.loads(path.read_text())
            key = data.get("_key", path.stem)
            result[key] = data
        return result

    def search(self, namespace: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        query_terms = query.lower().split()
        results: list[tuple[int, dict[str, Any]]] = []
        for _key, data in self.list_all(namespace).items():
            text = " ".join(str(v).lower() for v in data.values() if isinstance(v, str))
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                results.append((score, data))
        results.sort(key=lambda x: x[0], reverse=True)
        return [data for _, data in results[:limit]]

    def count(self, namespace: str) -> int:
        return len(list(self._ns_dir(namespace).glob("*.json")))

    def delete(self, namespace: str, key: str) -> bool:
        path = self._ns_dir(namespace) / f"{key}.json"
        if path.exists():
            path.unlink()
            return True
        return False
