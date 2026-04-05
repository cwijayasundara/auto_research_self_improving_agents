"""Abstract base classes that define the pluggable contracts.

Every higher-layer component programs against these protocols, not
concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from evoagent.core.types import GraderResult, TaskResult


class Grader(ABC):
    """Evaluate one axis of agent output quality."""

    name: str

    @abstractmethod
    def grade(self, task: str, output: str, **kwargs: Any) -> GraderResult: ...


class AgentFactory(ABC):
    """Build and invoke the agent under evaluation."""

    @abstractmethod
    def create(self, system_prompt: str, middleware: list[Any], **kwargs: Any) -> Any: ...

    @abstractmethod
    def run(self, agent: Any, task: str, timeout: int = 300) -> TaskResult: ...


class MemoryBackend(ABC):
    """Storage backend for episodic/semantic memories."""

    @abstractmethod
    def store(self, namespace: str, key: str, data: dict[str, Any]) -> None: ...

    @abstractmethod
    def retrieve(self, namespace: str, key: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def list_all(self, namespace: str) -> dict[str, dict[str, Any]]: ...

    @abstractmethod
    def search(self, namespace: str, query: str, limit: int = 10) -> list[dict[str, Any]]: ...

    @abstractmethod
    def count(self, namespace: str) -> int: ...

    @abstractmethod
    def delete(self, namespace: str, key: str) -> bool: ...


class SkillStore(ABC):
    """CRUD for skill files."""

    @abstractmethod
    def discover(self) -> dict[str, dict[str, Any]]: ...

    @abstractmethod
    def load(self, name: str) -> str | None: ...

    @abstractmethod
    def create(self, name: str, description: str, content: str) -> Path: ...


class PromptStore(ABC):
    """Versioned prompt persistence."""

    @abstractmethod
    def get_current(self) -> tuple[int, str]: ...

    @abstractmethod
    def save(self, content: str, score: float | None = None, parent: int | None = None) -> int: ...

    @abstractmethod
    def update_score(self, version: int, score: float) -> None: ...
