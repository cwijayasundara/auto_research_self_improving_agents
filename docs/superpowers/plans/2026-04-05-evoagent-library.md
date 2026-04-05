# EvoAgent Library Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the generic self-improving agent patterns from the research agent codebase into a reusable `evoagent` library with optional dependency layers.

**Architecture:** Single Python package with 4 layers (core → memory/skills → harness/graders/tracing → evolution). Each layer depends only on protocols defined in core. The current research agent becomes `examples/research_agent/`.

**Tech Stack:** Python 3.12+, Pydantic, LangChain, LangGraph, LangSmith

**Spec:** `docs/superpowers/specs/2026-04-05-evoagent-library-design.md`

---

## File Structure

### New Library Package (`evoagent/`)

```
evoagent/
├── __init__.py                    # Top-level convenience imports
├── core/
│   ├── __init__.py
│   ├── types.py                   # GraderResult, TaskResult, TrajectoryMetrics, EvolutionCycleReport
│   ├── protocols.py               # ABCs: Grader, AgentFactory, MemoryBackend, SkillStore, PromptStore, MiddlewareHook
│   ├── config.py                  # EvoAgentConfig base settings
│   └── parsing.py                 # Shared LLM JSON response parsing
├── memory/
│   ├── __init__.py
│   ├── store.py                   # FileMemoryStore (implements MemoryBackend)
│   ├── reflection.py              # Reflection generator (configurable prompt)
│   └── compression.py             # Token-budgeted context assembly
├── skills/
│   ├── __init__.py
│   ├── manager.py                 # SkillManager (implements SkillStore)
│   └── extractor.py               # Unified success + failure skill extraction
├── harness/
│   ├── __init__.py                # Import guard for langchain
│   ├── middleware.py              # 4 middleware classes (parameterized)
│   └── builder.py                 # default_middleware_stack() factory
├── graders/
│   ├── __init__.py                # Import guard for langchain
│   ├── multi_judge.py             # MultiJudgeGrader (implements Grader)
│   ├── efficiency.py              # EfficiencyGrader (implements Grader)
│   ├── claim_verify.py            # ClaimVerificationGrader (implements Grader)
│   └── fact_check.py              # FactCheckGrader (implements Grader)
├── evolution/
│   ├── __init__.py                # Import guard for langgraph
│   ├── loop.py                    # EvolutionLoop (the Karpathy outer loop)
│   ├── analyzer.py                # analyze_trajectory (accepts list[Grader])
│   ├── prompt_optimizer.py        # Metaprompt-based prompt rewriting
│   ├── state.py                   # Evolution state persistence
│   └── sleep_review.py            # Cross-run trace analysis
└── tracing/
    ├── __init__.py
    ├── capture.py                 # TraceCaptureMiddleware disk writer
    ├── fetcher.py                 # LangSmith trace fetching
    └── trajectory.py              # Trajectory data models
```

### Example Application (`examples/research_agent/`)

```
examples/research_agent/
├── __init__.py
├── agent.py                       # ResearchAgentFactory (implements AgentFactory)
├── prompts.py                     # Research-specific prompt templates
├── tools.py                       # ResilientSearch tool
├── subagents.py                   # Research + synthesis sub-agents
├── tasks.json                     # Sample evaluation tasks
├── main.py                        # CLI entry point using evoagent
└── README.md
```

### Test Structure

```
tests/
├── unit/
│   ├── test_types.py
│   ├── test_parsing.py
│   ├── test_config.py
│   ├── test_memory_store.py
│   ├── test_compression.py
│   ├── test_skill_manager.py
│   ├── test_skill_extractor.py
│   └── test_efficiency_grader.py
├── integration/
│   ├── test_multi_judge_grader.py
│   ├── test_claim_verification.py
│   ├── test_middleware.py
│   └── test_analyzer.py
└── e2e/
    └── test_evolution_loop.py
```

---

## Task 1: Scaffold the evoagent package and pyproject.toml

**Files:**
- Create: `evoagent/__init__.py`
- Create: `evoagent/core/__init__.py`
- Create: `evoagent/memory/__init__.py`
- Create: `evoagent/skills/__init__.py`
- Create: `evoagent/harness/__init__.py`
- Create: `evoagent/graders/__init__.py`
- Create: `evoagent/evolution/__init__.py`
- Create: `evoagent/tracing/__init__.py`
- Create: `evoagent/pyproject.toml`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p evoagent/{core,memory,skills,harness,graders,evolution,tracing}
```

- [ ] **Step 2: Create `evoagent/pyproject.toml`**

```toml
[project]
name = "evoagent"
version = "0.1.0"
description = "Composable self-improving agent framework: Karpathy-loop evolution, harness middleware, memory, grading, and skill learning"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
]

[project.optional-dependencies]
harness = [
    "langchain-core>=0.3",
]
graders = [
    "langchain-core>=0.3",
]
evolution = [
    "langchain-core>=0.3",
    "langgraph>=1.0",
    "langsmith>=0.7",
]
all = [
    "evoagent[harness,graders,evolution]",
]
dev = [
    "pytest>=9.0",
    "pytest-cov>=7.0",
    "ruff>=0.15",
    "mypy>=1.19",
]

[tool.ruff]
target-version = "py312"
line-length = 100
src = ["evoagent", "tests"]

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "SIM", "RUF"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["evoagent"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "unit: Unit tests",
    "integration: Integration tests with mocked LLM",
    "e2e: End-to-end tests",
]
addopts = ["--strict-markers", "--tb=short", "-q"]
```

- [ ] **Step 3: Create `evoagent/__init__.py`**

```python
"""EvoAgent: Composable self-improving agent framework."""

from evoagent.core.config import EvoAgentConfig
from evoagent.core.types import (
    EvolutionCycleReport,
    GraderResult,
    TaskResult,
    TrajectoryMetrics,
)

__all__ = [
    "EvoAgentConfig",
    "EvolutionCycleReport",
    "GraderResult",
    "TaskResult",
    "TrajectoryMetrics",
]
```

- [ ] **Step 4: Create `evoagent/core/__init__.py`**

```python
"""Core types, protocols, and configuration — no heavy dependencies."""
```

- [ ] **Step 5: Create layer `__init__.py` files with import guards**

`evoagent/memory/__init__.py`:
```python
"""Episodic/semantic memory system."""
```

`evoagent/skills/__init__.py`:
```python
"""SKILL.md management and extraction."""
```

`evoagent/tracing/__init__.py`:
```python
"""Trace capture and trajectory models."""
```

`evoagent/harness/__init__.py`:
```python
"""Harness middleware stack for LangChain agents."""

try:
    from langchain_core.messages import BaseMessage as _  # noqa: F401
except ImportError:
    raise ImportError(
        "evoagent.harness requires langchain-core. "
        "Install with: pip install evoagent[harness]"
    ) from None
```

`evoagent/graders/__init__.py`:
```python
"""Pluggable multi-axis grading system."""

try:
    from langchain_core.messages import BaseMessage as _  # noqa: F401
except ImportError:
    raise ImportError(
        "evoagent.graders requires langchain-core. "
        "Install with: pip install evoagent[graders]"
    ) from None
```

`evoagent/evolution/__init__.py`:
```python
"""Evolution loop: propose -> run -> grade -> accept/reject."""

try:
    from langgraph.graph import StateGraph as _  # noqa: F401
except ImportError:
    raise ImportError(
        "evoagent.evolution requires langgraph. "
        "Install with: pip install evoagent[evolution]"
    ) from None
```

- [ ] **Step 6: Commit**

```bash
git add evoagent/ evoagent/pyproject.toml
git commit -m "feat(evoagent): scaffold package structure with layered import guards"
```

---

## Task 2: Core types and parsing utilities

**Files:**
- Create: `evoagent/core/types.py`
- Create: `evoagent/core/parsing.py`
- Create: `tests/unit/test_types.py`
- Create: `tests/unit/test_parsing.py`

- [ ] **Step 1: Write failing tests for types**

Create `tests/unit/test_types.py`:

```python
"""Tests for evoagent core types."""

from evoagent.core.types import GraderResult, TaskResult, TrajectoryMetrics, EvolutionCycleReport


def test_grader_result_fields():
    r = GraderResult(name="quality", score=0.85, passed=True, reasoning="Good structure")
    assert r.name == "quality"
    assert r.score == 0.85
    assert r.passed is True
    assert r.reasoning == "Good structure"


def test_task_result_fields():
    r = TaskResult(
        task="Research quantum computing",
        output="Report content...",
        tool_calls=[{"name": "search", "args": {"q": "quantum"}}],
        duration_seconds=25.0,
        status="success",
    )
    assert r.status == "success"
    assert r.duration_seconds == 25.0


def test_trajectory_metrics_defaults():
    m = TrajectoryMetrics()
    assert m.total_tokens == 0
    assert m.total_steps == 0
    assert m.latency_seconds == 0.0
    assert m.tool_call_count == 0


def test_evolution_cycle_report():
    r = EvolutionCycleReport(
        cycle=3,
        avg_score=0.82,
        classification_counts={"successful": 2, "partial": 1, "failed": 0},
        prompt_version=5,
        skills_created=2,
        improvements=["Better error handling"],
    )
    assert r.cycle == 3
    assert r.avg_score == 0.82
```

- [ ] **Step 2: Write failing tests for parsing**

Create `tests/unit/test_parsing.py`:

```python
"""Tests for LLM JSON response parsing."""

from evoagent.core.parsing import parse_llm_json


def test_parse_plain_json():
    assert parse_llm_json('{"score": 0.8}') == {"score": 0.8}


def test_parse_markdown_json_block():
    raw = '```json\n{"score": 0.8, "reasoning": "good"}\n```'
    result = parse_llm_json(raw)
    assert result == {"score": 0.8, "reasoning": "good"}


def test_parse_markdown_block_no_lang():
    raw = '```\n{"score": 0.5}\n```'
    assert parse_llm_json(raw) == {"score": 0.5}


def test_parse_invalid_json_returns_empty():
    assert parse_llm_json("not json at all") == {}


def test_parse_empty_string():
    assert parse_llm_json("") == {}


def test_parse_with_trailing_text():
    raw = '```json\n{"score": 0.9}\n```\nSome trailing text'
    result = parse_llm_json(raw)
    assert result["score"] == 0.9
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd evoagent && python -m pytest tests/unit/test_types.py tests/unit/test_parsing.py -v
```

Expected: FAIL — modules don't exist yet.

- [ ] **Step 4: Implement `evoagent/core/types.py`**

```python
"""Shared types for the evoagent framework.

All types are plain dataclasses with no heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GraderResult:
    """Result from a single grader evaluation."""

    name: str
    score: float  # 0.0 - 1.0
    passed: bool
    reasoning: str


@dataclass
class TaskResult:
    """Result from running the agent on a single task."""

    task: str
    output: str
    tool_calls: list[dict] = field(default_factory=list)
    duration_seconds: float = 0.0
    status: Literal["success", "error", "timeout"] = "success"


@dataclass
class TrajectoryMetrics:
    """Efficiency metrics for a single agent run."""

    total_tokens: int = 0
    total_steps: int = 0
    latency_seconds: float = 0.0
    tool_call_count: int = 0


@dataclass
class EvolutionCycleReport:
    """Summary of a single evolution cycle."""

    cycle: int
    avg_score: float
    classification_counts: dict[str, int] = field(default_factory=dict)
    prompt_version: int = 0
    skills_created: int = 0
    improvements: list[str] = field(default_factory=list)
```

- [ ] **Step 5: Implement `evoagent/core/parsing.py`**

```python
"""Shared LLM JSON response parsing utilities.

Handles markdown code blocks, trailing text, and graceful fallback.
Extracted from duplicated logic across multi_judge.py, skill_extractor.py,
failure_skill_creator.py, and sleep_review.py.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_llm_json(text: str) -> dict[str, Any]:
    """Parse JSON from an LLM response, handling markdown code blocks.

    Tries in order:
    1. Strip markdown ```json ... ``` wrapper
    2. Parse as raw JSON
    3. Return empty dict on failure
    """
    cleaned = text.strip()
    if not cleaned:
        return {}

    # Strip markdown code block wrapper
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```) and any closing ```
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        logger.debug("Failed to parse LLM JSON: %s", cleaned[:200])
        return {}
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd evoagent && python -m pytest tests/unit/test_types.py tests/unit/test_parsing.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add evoagent/core/types.py evoagent/core/parsing.py tests/unit/test_types.py tests/unit/test_parsing.py
git commit -m "feat(evoagent): add core types and LLM JSON parsing utilities"
```

---

## Task 3: Core protocols

**Files:**
- Create: `evoagent/core/protocols.py`

- [ ] **Step 1: Implement `evoagent/core/protocols.py`**

```python
"""Abstract base classes that define the pluggable contracts.

Every higher-layer component programs against these protocols, not
concrete implementations. Users implement the protocols they want to
customize and pass them to the library.
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
    def grade(self, task: str, output: str, **kwargs: Any) -> GraderResult:
        """Score the agent's output on this grader's axis.

        Args:
            task: The original task description.
            output: The agent's output text.
            **kwargs: Extra context (e.g., metrics, tool_calls).

        Returns:
            GraderResult with score 0.0-1.0.
        """


class AgentFactory(ABC):
    """Build and invoke the agent under evaluation."""

    @abstractmethod
    def create(self, system_prompt: str, middleware: list[Any], **kwargs: Any) -> Any:
        """Create an agent instance with the given prompt and middleware."""

    @abstractmethod
    def run(self, agent: Any, task: str, timeout: int = 300) -> TaskResult:
        """Run the agent on a task and return structured results."""


class MemoryBackend(ABC):
    """Storage backend for episodic/semantic memories."""

    @abstractmethod
    def store(self, namespace: str, key: str, data: dict[str, Any]) -> None:
        """Store a memory document."""

    @abstractmethod
    def retrieve(self, namespace: str, key: str) -> dict[str, Any] | None:
        """Retrieve a specific memory by key."""

    @abstractmethod
    def list_all(self, namespace: str) -> dict[str, dict[str, Any]]:
        """List all memories in a namespace. Returns {key: data}."""

    @abstractmethod
    def search(self, namespace: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search memories by keyword matching."""

    @abstractmethod
    def count(self, namespace: str) -> int:
        """Count memories in a namespace."""

    @abstractmethod
    def delete(self, namespace: str, key: str) -> bool:
        """Delete a specific memory. Returns True if deleted."""


class SkillStore(ABC):
    """CRUD for skill files."""

    @abstractmethod
    def discover(self) -> dict[str, dict[str, Any]]:
        """Discover all skills. Returns {skill_id: {name, description, path}}."""

    @abstractmethod
    def load(self, name: str) -> str | None:
        """Load full skill content by name. Returns None if not found."""

    @abstractmethod
    def create(self, name: str, description: str, content: str) -> Path:
        """Create a new skill. Returns path to the created file."""


class PromptStore(ABC):
    """Versioned prompt persistence."""

    @abstractmethod
    def get_current(self) -> tuple[int, str]:
        """Get the best prompt. Returns (version_number, prompt_text)."""

    @abstractmethod
    def save(self, content: str, score: float | None = None, parent: int | None = None) -> int:
        """Save a new prompt version. Returns the new version number."""

    @abstractmethod
    def update_score(self, version: int, score: float) -> None:
        """Update the score for an existing prompt version."""
```

- [ ] **Step 2: Commit**

```bash
git add evoagent/core/protocols.py
git commit -m "feat(evoagent): add core protocol ABCs for pluggable components"
```

---

## Task 4: Core config

**Files:**
- Create: `evoagent/core/config.py`
- Create: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_config.py`:

```python
"""Tests for EvoAgentConfig."""

from pathlib import Path

from evoagent.core.config import EvoAgentConfig


def test_default_config():
    config = EvoAgentConfig()
    assert config.max_cycles == 5
    assert config.min_improvement_threshold == 0.05
    assert config.memory_token_budget == 4000


def test_config_override():
    config = EvoAgentConfig(max_cycles=20, memory_token_budget=8000)
    assert config.max_cycles == 20
    assert config.memory_token_budget == 8000


def test_config_paths(tmp_path):
    config = EvoAgentConfig(base_dir=str(tmp_path))
    assert config.memory_path == tmp_path / "memory"
    assert config.skills_path == tmp_path / "skills"
    assert config.prompts_path == tmp_path / "prompts"
    assert config.traces_path == tmp_path / "traces"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd evoagent && python -m pytest tests/unit/test_config.py -v
```

- [ ] **Step 3: Implement `evoagent/core/config.py`**

```python
"""Base configuration for EvoAgent.

Provides sensible defaults. Users can subclass or override via env vars.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class EvoAgentConfig(BaseSettings):
    """Configuration for the evoagent framework."""

    model_config = {"extra": "ignore"}

    # Base directory for all storage (memory, skills, prompts, traces)
    base_dir: str = "."

    # Evolution loop
    max_cycles: int = 5
    batch_size: int = 3
    min_improvement_threshold: float = 0.05
    plateau_cycles: int = 2
    task_timeout_seconds: int = 300

    # Memory
    memory_token_budget: int = 4000
    compression_similarity_threshold: float = 0.7

    # Directory names (relative to base_dir)
    memory_dir: str = "memory"
    skills_dir: str = "skills"
    prompts_dir: str = "prompts"
    traces_dir: str = "traces"
    evolution_state_dir: str = "evolution_state"

    @property
    def _base(self) -> Path:
        return Path(self.base_dir)

    @property
    def memory_path(self) -> Path:
        return self._base / self.memory_dir

    @property
    def skills_path(self) -> Path:
        return self._base / self.skills_dir

    @property
    def prompts_path(self) -> Path:
        return self._base / self.prompts_dir

    @property
    def traces_path(self) -> Path:
        return self._base / self.traces_dir

    @property
    def evolution_state_path(self) -> Path:
        return self._base / self.evolution_state_dir
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd evoagent && python -m pytest tests/unit/test_config.py -v
```

- [ ] **Step 5: Commit**

```bash
git add evoagent/core/config.py tests/unit/test_config.py
git commit -m "feat(evoagent): add EvoAgentConfig with path properties"
```

---

## Task 5: Memory system (store + compression)

**Files:**
- Create: `evoagent/memory/store.py`
- Create: `evoagent/memory/compression.py`
- Create: `tests/unit/test_memory_store.py`
- Create: `tests/unit/test_compression.py`

- [ ] **Step 1: Write failing tests for memory store**

Create `tests/unit/test_memory_store.py`:

```python
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
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd evoagent && python -m pytest tests/unit/test_memory_store.py -v
```

- [ ] **Step 3: Implement `evoagent/memory/store.py`**

Adapt from `src/memory/store.py`. Key change: implement `MemoryBackend` protocol, rename class to `FileMemoryStore`, and change `list_all` to return `dict[str, dict]` instead of `list[dict]`.

```python
"""File-backed memory store implementing MemoryBackend protocol.

Supports two namespaces: episodic (past run experiences) and
semantic (learned facts and patterns).
"""

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
        logger.info("Stored %s memory: %s", namespace, key)

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
            logger.info("Deleted %s memory: %s", namespace, key)
            return True
        return False
```

- [ ] **Step 4: Run memory tests**

```bash
cd evoagent && python -m pytest tests/unit/test_memory_store.py -v
```

Expected: all PASS.

- [ ] **Step 5: Write failing tests for compression**

Create `tests/unit/test_compression.py`:

```python
"""Tests for memory compression."""

from evoagent.memory.store import FileMemoryStore
from evoagent.memory.compression import compress_context, deduplicate_semantic


def test_compress_empty_memory(tmp_path):
    store = FileMemoryStore(tmp_path)
    context = compress_context(store, task="test", token_budget=1000)
    assert isinstance(context, str)


def test_compress_respects_budget(tmp_path):
    store = FileMemoryStore(tmp_path)
    for i in range(20):
        store.store("semantic", f"fact-{i}", {"content": f"Fact number {i} " * 50})
    context = compress_context(store, task="test", token_budget=500)
    # ~4 chars per token, 500 tokens ~ 2000 chars
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
```

- [ ] **Step 6: Implement `evoagent/memory/compression.py`**

Adapt from `src/memory/compression.py`. Key change: accept `MemoryBackend` instead of `MemoryStore`. The `list_all` method now returns `dict[str, dict]`.

```python
"""Token-budgeted memory context assembly and deduplication.

Assembles relevant memories into a context string that fits within
a token budget. Prioritizes meta-instructions from sleep-time review.
"""

from __future__ import annotations

import logging
from typing import Any

from evoagent.core.protocols import MemoryBackend

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4


def compress_context(
    memory: MemoryBackend,
    task: str,
    token_budget: int = 4000,
) -> str:
    """Assemble relevant memories into a token-budgeted context string.

    Priority order:
    1. Meta-instructions (from sleep-time review)
    2. Semantic memories matching the task
    3. Recent episodic memories
    """
    char_budget = token_budget * CHARS_PER_TOKEN
    semantic_items = list(memory.list_all("semantic").values())
    episodic_items = list(memory.list_all("episodic").values())

    parts: list[str] = ["\n\n## Relevant Past Experience"]
    current_len = len(parts[0])

    # Meta-instructions first
    meta = [m for m in semantic_items if m.get("type") == "meta_instruction"]
    regular = [m for m in semantic_items if m.get("type") != "meta_instruction"]

    if meta:
        header = "### Strategy Guidelines (from cross-run analysis)"
        current_len += len(header) + 1
        if current_len < char_budget:
            parts.append(header)
            for mem in meta:
                line = f"- {mem.get('content', str(mem))}"
                if current_len + len(line) + 1 > char_budget:
                    break
                parts.append(line)
                current_len += len(line) + 1

    if regular:
        header = "### Learned Facts & Patterns"
        current_len += len(header) + 1
        if current_len < char_budget:
            parts.append(header)
            for mem in regular:
                line = f"- {mem.get('content', str(mem))}"
                if current_len + len(line) + 1 > char_budget:
                    break
                parts.append(line)
                current_len += len(line) + 1

    if episodic_items:
        header = "### Recent Run Summaries"
        current_len += len(header) + 1
        if current_len < char_budget:
            parts.append(header)
            for mem in episodic_items[-10:]:
                task_str = mem.get("task", "?")[:80]
                score = mem.get("score", "?")
                summary = mem.get("summary", "")[:100]
                line = f"- [{score}] {task_str}: {summary}"
                if current_len + len(line) + 1 > char_budget:
                    break
                parts.append(line)
                current_len += len(line) + 1

    return "\n".join(parts) if len(parts) > 1 else ""


def _jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def deduplicate_semantic(
    memory: MemoryBackend,
    threshold: float = 0.7,
) -> int:
    """Remove near-duplicate semantic memories.

    Returns the count of removed memories.
    """
    items = memory.list_all("semantic")
    keys = list(items.keys())
    to_remove: set[str] = set()

    for i in range(len(keys)):
        if keys[i] in to_remove:
            continue
        content_i = items[keys[i]].get("content", str(items[keys[i]]))
        for j in range(i + 1, len(keys)):
            if keys[j] in to_remove:
                continue
            content_j = items[keys[j]].get("content", str(items[keys[j]]))
            if _jaccard_similarity(content_i, content_j) >= threshold:
                to_remove.add(keys[j])

    for key in to_remove:
        memory.delete("semantic", key)

    if to_remove:
        logger.info("Deduplicated %d semantic memories", len(to_remove))
    return len(to_remove)
```

- [ ] **Step 7: Run all memory + compression tests**

```bash
cd evoagent && python -m pytest tests/unit/test_memory_store.py tests/unit/test_compression.py -v
```

Expected: all PASS.

- [ ] **Step 8: Update `evoagent/memory/__init__.py` exports**

```python
"""Episodic/semantic memory system."""

from evoagent.memory.compression import compress_context, deduplicate_semantic
from evoagent.memory.store import FileMemoryStore

__all__ = ["FileMemoryStore", "compress_context", "deduplicate_semantic"]
```

- [ ] **Step 9: Commit**

```bash
git add evoagent/memory/ tests/unit/test_memory_store.py tests/unit/test_compression.py
git commit -m "feat(evoagent): add memory system with FileMemoryStore and compression"
```

---

## Task 6: Skills system (manager + extractor)

**Files:**
- Create: `evoagent/skills/manager.py`
- Create: `evoagent/skills/extractor.py`
- Create: `tests/unit/test_skill_manager.py`
- Create: `tests/unit/test_skill_extractor.py`

- [ ] **Step 1: Write failing tests for skill manager**

Create `tests/unit/test_skill_manager.py`:

```python
"""Tests for SkillManager."""

from evoagent.skills.manager import SkillManager


def test_discover_empty(tmp_path):
    mgr = SkillManager(tmp_path)
    assert mgr.discover() == {}


def test_create_and_discover(tmp_path):
    mgr = SkillManager(tmp_path)
    path = mgr.create("retry-search", "Retry failed searches", "# Instructions\nRetry with backoff")
    assert path.exists()
    skills = mgr.discover()
    assert "retry-search" in skills
    assert skills["retry-search"]["name"] == "retry-search"


def test_load_skill(tmp_path):
    mgr = SkillManager(tmp_path)
    mgr.create("my-skill", "Test skill", "# Steps\n1. Do thing")
    content = mgr.load("my-skill")
    assert content is not None
    assert "Do thing" in content


def test_load_missing_skill(tmp_path):
    mgr = SkillManager(tmp_path)
    assert mgr.load("nonexistent") is None


def test_validate_skill(tmp_path):
    mgr = SkillManager(tmp_path)
    path = mgr.create("valid", "A valid skill", "# Content\nSome body text")
    is_valid, msg = mgr.validate(path)
    assert is_valid
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd evoagent && python -m pytest tests/unit/test_skill_manager.py -v
```

- [ ] **Step 3: Implement `evoagent/skills/manager.py`**

Adapt from `src/skills/manager.py`. Key change: wrap as a class implementing `SkillStore`.

```python
"""SKILL.md management implementing SkillStore protocol.

Skills use Anthropic's skill-creator convention:
---
name: skill-name
description: What it does and when to use it
---
# Instructions
...
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from evoagent.core.protocols import SkillStore

logger = logging.getLogger(__name__)


def parse_frontmatter(content: str) -> dict[str, Any]:
    """Parse YAML frontmatter from SKILL.md content."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
    if not match:
        return {"name": "unknown", "description": "No description", "body": content}

    metadata: dict[str, str] = {}
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()

    return {
        "name": metadata.get("name", "unknown"),
        "description": metadata.get("description", "No description"),
        "body": match.group(2).strip(),
    }


class SkillManager(SkillStore):
    """File-backed SKILL.md CRUD with progressive disclosure."""

    def __init__(self, skills_dir: Path | str) -> None:
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def discover(self) -> dict[str, dict[str, Any]]:
        """Discover all skills (name + description only, progressive disclosure)."""
        skills: dict[str, dict[str, Any]] = {}
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    parsed = parse_frontmatter(skill_file.read_text())
                    skills[skill_dir.name] = {
                        "name": parsed["name"],
                        "description": parsed["description"],
                        "path": str(skill_file),
                    }
        return skills

    def load(self, name: str) -> str | None:
        """Load full skill content by ID."""
        skill_file = self.skills_dir / name / "SKILL.md"
        if not skill_file.exists():
            return None
        parsed = parse_frontmatter(skill_file.read_text())
        return parsed["body"]

    def create(self, name: str, description: str, content: str) -> Path:
        """Create a new skill as a SKILL.md file."""
        skill_id = name.lower().replace(" ", "-").replace("_", "-")
        skill_dir = self.skills_dir / skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n{content}\n")
        logger.info("Created skill '%s' at %s", name, skill_file)
        return skill_file

    def validate(self, skill_path: Path) -> tuple[bool, str]:
        """Validate a SKILL.md file structure."""
        if not skill_path.exists():
            return False, f"File not found: {skill_path}"
        content = skill_path.read_text()
        if not content.startswith("---"):
            return False, "Missing YAML frontmatter"
        parsed = parse_frontmatter(content)
        if parsed["name"] == "unknown":
            return False, "Missing 'name' in frontmatter"
        if parsed["description"] == "No description":
            return False, "Missing 'description' in frontmatter"
        if not parsed["body"].strip():
            return False, "Skill body is empty"
        return True, "Valid"
```

- [ ] **Step 4: Run tests**

```bash
cd evoagent && python -m pytest tests/unit/test_skill_manager.py -v
```

Expected: all PASS.

- [ ] **Step 5: Write failing tests for skill extractor**

Create `tests/unit/test_skill_extractor.py`:

```python
"""Tests for unified skill extraction (success + failure)."""

from unittest.mock import MagicMock

from evoagent.core.types import GraderResult
from evoagent.skills.extractor import is_duplicate_skill, extract_skills_from_batch
from evoagent.skills.manager import SkillManager


def test_is_duplicate_exact_match():
    existing = {"retry-search": {}, "handle-timeout": {}}
    assert is_duplicate_skill("retry-search", existing) is True


def test_is_duplicate_word_overlap():
    existing = {"retry-search-queries": {}}
    assert is_duplicate_skill("retry-search", existing) is True


def test_not_duplicate():
    existing = {"retry-search": {}}
    assert is_duplicate_skill("handle-timeout", existing) is False


def test_extract_skips_low_score(tmp_path):
    mgr = SkillManager(tmp_path)
    analyses = [
        {
            "run_id": "r1",
            "task": "test",
            "classification": "failed",
            "average_score": 0.3,
            "grader_results": [],
            "output": "bad output",
            "tool_calls": [],
        }
    ]
    # No LLM call needed — below threshold
    created = extract_skills_from_batch(
        llm=MagicMock(), analyses=analyses, skill_store=mgr, mode="success"
    )
    assert created == []
```

- [ ] **Step 6: Implement `evoagent/skills/extractor.py`**

Unified extraction for both success and failure skills. The LLM prompt is a constructor parameter.

```python
"""Unified skill extraction from successful and failed trajectories.

Merges the logic from src/evolution/skill_extractor.py and
src/evolution/failure_skill_creator.py into a single module.
The extraction prompt is configurable per mode (success vs failure).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from evoagent.core.parsing import parse_llm_json
from evoagent.skills.manager import SkillManager

logger = logging.getLogger(__name__)

SUCCESS_THRESHOLD = 0.70
FAILURE_THRESHOLD = 0.50
MIN_FAILURES_FOR_SKILL = 2
MAX_SKILLS_PER_CYCLE = 3

DEFAULT_SUCCESS_PROMPT = (
    "Analyze this successful agent trajectory and extract a reusable skill.\n\n"
    "Task: {task}\nOutput (preview): {output}\nScore: {score}\n\n"
    "Respond as JSON: {{\"name\": \"...\", \"description\": \"...\", \"content\": \"...\"}}"
)

DEFAULT_FAILURE_PROMPT = (
    "Analyze these failed trajectories and create a defensive skill.\n\n"
    "Failures:\n{failures}\n\nFailure patterns:\n{patterns}\n\n"
    "Respond as JSON: {{\"name\": \"...\", \"description\": \"...\", \"content\": \"...\"}}"
)


def is_duplicate_skill(name: str, existing_skills: dict[str, Any]) -> bool:
    """Check if a skill with a similar name already exists."""
    normalized = name.lower().replace(" ", "-").replace("_", "-")
    for skill_id in existing_skills:
        if skill_id == normalized:
            return True
        existing_words = set(skill_id.split("-"))
        new_words = set(normalized.split("-"))
        if len(existing_words & new_words) >= 2:
            return True
    return False


def extract_skills_from_batch(
    llm: Any,
    analyses: list[dict[str, Any]],
    skill_store: SkillManager,
    mode: str = "success",
    prompt_template: str | None = None,
) -> list[Path]:
    """Extract skills from a batch of analyses.

    Args:
        llm: Language model for generating skill content.
        analyses: List of analysis result dicts.
        skill_store: Where to store created skills.
        mode: "success" extracts from high-score runs, "failure" from low-score.
        prompt_template: Custom extraction prompt (uses default if None).

    Returns:
        List of paths to created skill files.
    """
    from langchain_core.messages import HumanMessage

    if mode == "success":
        threshold = SUCCESS_THRESHOLD
        eligible = [a for a in analyses if a.get("average_score", 0) >= threshold]
        template = prompt_template or DEFAULT_SUCCESS_PROMPT
    else:
        threshold = FAILURE_THRESHOLD
        eligible = [a for a in analyses if a.get("average_score", 0) < threshold]
        template = prompt_template or DEFAULT_FAILURE_PROMPT

    if not eligible:
        return []

    existing = skill_store.discover()
    created: list[Path] = []

    for analysis in eligible[:MAX_SKILLS_PER_CYCLE]:
        try:
            if mode == "success":
                prompt = template.format(
                    task=analysis.get("task", ""),
                    output=analysis.get("output", "")[:3000],
                    score=analysis.get("average_score", 0),
                )
            else:
                prompt = template.format(
                    failures=analysis.get("output", "")[:2000],
                    patterns=str(analysis.get("grader_results", []))[:1000],
                )

            response = llm.invoke([HumanMessage(content=prompt)])
            parsed = parse_llm_json(response.content)

            if not parsed.get("name") or not parsed.get("content"):
                continue

            if is_duplicate_skill(parsed["name"], existing):
                logger.info("Skill '%s' is duplicate, skipping", parsed["name"])
                continue

            path = skill_store.create(
                name=parsed["name"],
                description=parsed.get("description", f"Extracted from {mode} run"),
                content=parsed["content"],
            )
            is_valid, msg = skill_store.validate(path)
            if not is_valid:
                path.unlink()
                continue

            created.append(path)
            existing[parsed["name"].lower().replace(" ", "-")] = {}

        except Exception as exc:
            logger.error("Skill extraction failed: %s", exc)

    return created
```

- [ ] **Step 7: Run skill tests**

```bash
cd evoagent && python -m pytest tests/unit/test_skill_manager.py tests/unit/test_skill_extractor.py -v
```

Expected: all PASS.

- [ ] **Step 8: Update `evoagent/skills/__init__.py`**

```python
"""SKILL.md management and extraction."""

from evoagent.skills.extractor import extract_skills_from_batch
from evoagent.skills.manager import SkillManager

__all__ = ["SkillManager", "extract_skills_from_batch"]
```

- [ ] **Step 9: Commit**

```bash
git add evoagent/skills/ tests/unit/test_skill_manager.py tests/unit/test_skill_extractor.py
git commit -m "feat(evoagent): add skills system with SkillManager and unified extractor"
```

---

## Task 7: Graders (efficiency + multi-judge)

**Files:**
- Create: `evoagent/graders/efficiency.py`
- Create: `evoagent/graders/multi_judge.py`
- Create: `tests/unit/test_efficiency_grader.py`
- Create: `tests/integration/test_multi_judge_grader.py`

- [ ] **Step 1: Write failing tests for efficiency grader**

Create `tests/unit/test_efficiency_grader.py`:

```python
"""Tests for EfficiencyGrader."""

from evoagent.core.types import TrajectoryMetrics
from evoagent.graders.efficiency import EfficiencyGrader


def test_ideal_metrics():
    grader = EfficiencyGrader()
    result = grader.grade(
        task="test",
        output="test output",
        metrics=TrajectoryMetrics(total_tokens=5000, total_steps=3, latency_seconds=15.0),
    )
    assert result.score > 0.8
    assert result.passed is True
    assert result.name == "efficiency"


def test_bad_metrics():
    grader = EfficiencyGrader()
    result = grader.grade(
        task="test",
        output="test output",
        metrics=TrajectoryMetrics(total_tokens=100000, total_steps=50, latency_seconds=300.0),
    )
    assert result.score < 0.5
    assert result.passed is False


def test_custom_thresholds():
    grader = EfficiencyGrader(
        max_ideal_tokens=1000,
        max_acceptable_tokens=5000,
    )
    result = grader.grade(
        task="test",
        output="test output",
        metrics=TrajectoryMetrics(total_tokens=3000),
    )
    assert result.score < 1.0  # Over ideal but under acceptable
```

- [ ] **Step 2: Implement `evoagent/graders/efficiency.py`**

```python
"""Rule-based efficiency grader implementing Grader protocol.

Scores agent efficiency based on token usage, step count, and latency.
No LLM calls required — purely deterministic.
"""

from __future__ import annotations

from evoagent.core.protocols import Grader
from evoagent.core.types import GraderResult, TrajectoryMetrics


class EfficiencyGrader(Grader):
    """Grade agent efficiency based on resource usage."""

    name = "efficiency"

    def __init__(
        self,
        max_ideal_tokens: int = 10_000,
        max_acceptable_tokens: int = 50_000,
        max_ideal_steps: int = 5,
        max_acceptable_steps: int = 15,
        max_ideal_latency: float = 30.0,
        max_acceptable_latency: float = 120.0,
        weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    ) -> None:
        self._ideal_tokens = max_ideal_tokens
        self._accept_tokens = max_acceptable_tokens
        self._ideal_steps = max_ideal_steps
        self._accept_steps = max_acceptable_steps
        self._ideal_latency = max_ideal_latency
        self._accept_latency = max_acceptable_latency
        self._weights = weights

    def grade(self, task: str, output: str, **kwargs) -> GraderResult:
        metrics: TrajectoryMetrics = kwargs.get("metrics", TrajectoryMetrics())

        token_score = _score_metric(metrics.total_tokens, self._ideal_tokens, self._accept_tokens)
        step_score = _score_metric(metrics.total_steps, self._ideal_steps, self._accept_steps)
        latency_score = _score_metric(
            metrics.latency_seconds, self._ideal_latency, self._accept_latency
        )

        w_tok, w_step, w_lat = self._weights
        score = w_tok * token_score + w_step * step_score + w_lat * latency_score

        reasoning = (
            f"Tokens: {metrics.total_tokens} ({token_score:.2f}); "
            f"Steps: {metrics.total_steps} ({step_score:.2f}); "
            f"Latency: {metrics.latency_seconds:.1f}s ({latency_score:.2f})"
        )

        return GraderResult(
            name=self.name,
            score=round(score, 3),
            passed=score >= 0.5,
            reasoning=reasoning,
        )


def _score_metric(value: float, ideal: float, acceptable: float) -> float:
    """Score a metric: ideal=1.0, acceptable=0.5, beyond=approaching 0."""
    if value <= ideal:
        return 1.0
    if value <= acceptable:
        return 1.0 - 0.5 * ((value - ideal) / (acceptable - ideal))
    return max(0.0, 0.5 - 0.5 * ((value - acceptable) / acceptable))
```

- [ ] **Step 3: Run efficiency tests**

```bash
cd evoagent && python -m pytest tests/unit/test_efficiency_grader.py -v
```

Expected: all PASS.

- [ ] **Step 4: Write failing tests for multi-judge grader**

Create `tests/integration/test_multi_judge_grader.py`:

```python
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
    assert result.score == 0.8  # median
    assert result.name == "quality"


def test_fallback_on_judge_failure():
    """When judges fail, falls back to single prompt."""
    llm = MagicMock()
    llm.invoke.side_effect = [
        Exception("timeout"),
        Exception("timeout"),
        Exception("timeout"),
        AIMessage(content='{"score": 0.7, "reasoning": "fallback"}'),  # fallback
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
```

- [ ] **Step 5: Implement `evoagent/graders/multi_judge.py`**

```python
"""Multi-judge grading implementing Grader protocol.

Runs 3 perspective judges in parallel, aggregates via median,
falls back to single judge if too many fail.
"""

from __future__ import annotations

import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evoagent.core.parsing import parse_llm_json
from evoagent.core.protocols import Grader
from evoagent.core.types import GraderResult

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_PROMPT = (
    "Rate the following agent output on a scale of 0.0 to 1.0.\n\n"
    "Task: {task}\n\nOutput:\n{output}\n\n"
    "Respond as JSON: {{\"score\": <float>, \"reasoning\": \"<string>\"}}"
)


class MultiJudgeGrader(Grader):
    """Run multiple LLM judges in parallel and aggregate via median."""

    def __init__(
        self,
        llm: BaseChatModel,
        name: str = "quality",
        judge_prompts: list[str] | None = None,
        fallback_prompt: str | None = None,
        pass_threshold: float = 0.75,
    ) -> None:
        self.name = name
        self._llm = llm
        self._judge_prompts = judge_prompts or [DEFAULT_JUDGE_PROMPT] * 3
        self._fallback_prompt = fallback_prompt or DEFAULT_JUDGE_PROMPT
        self._pass_threshold = pass_threshold

    def grade(self, task: str, output: str, **kwargs: Any) -> GraderResult:
        score, reasoning = self._run_multi_judge(task, output)
        return GraderResult(
            name=self.name,
            score=score,
            passed=score >= self._pass_threshold,
            reasoning=reasoning,
        )

    def _run_single_judge(self, prompt_template: str, task: str, output: str) -> tuple[float | None, str]:
        try:
            prompt = prompt_template.replace("{task}", task).replace("{output}", output[:4000])
            response = self._llm.invoke([HumanMessage(content=prompt)])
            parsed = parse_llm_json(response.content)
            if "score" not in parsed:
                return 0.5, f"parse_error: no score key in {response.content[:200]}"
            return float(parsed["score"]), parsed.get("reasoning", "")
        except Exception as exc:
            return None, str(exc)

    def _run_multi_judge(self, task: str, output: str) -> tuple[float, str]:
        scores: list[float] = []
        reasonings: list[str] = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._run_single_judge, p, task, output): i
                for i, p in enumerate(self._judge_prompts)
            }
            for future in as_completed(futures):
                score, reasoning = future.result()
                if score is not None:
                    scores.append(score)
                    reasonings.append(reasoning)

        if len(scores) >= 2:
            median_score = statistics.median(scores)
            combined = " | ".join(reasonings)
            spread = max(scores) - min(scores)
            if spread > 0.3:
                combined = f"[low_agreement spread={spread:.2f}] {combined}"
            return median_score, combined

        # Fallback
        logger.warning("Only %d/%d judges succeeded, falling back", len(scores), len(self._judge_prompts))
        score, reasoning = self._run_single_judge(self._fallback_prompt, task, output)
        if score is not None:
            return score, reasoning
        return 0.5, "grader_error: all judges and fallback failed"
```

- [ ] **Step 6: Run multi-judge tests**

```bash
cd evoagent && python -m pytest tests/integration/test_multi_judge_grader.py -v
```

Expected: all PASS.

- [ ] **Step 7: Update `evoagent/graders/__init__.py`**

```python
"""Pluggable multi-axis grading system."""

try:
    from langchain_core.messages import BaseMessage as _  # noqa: F401
except ImportError:
    raise ImportError(
        "evoagent.graders requires langchain-core. "
        "Install with: pip install evoagent[graders]"
    ) from None

from evoagent.graders.efficiency import EfficiencyGrader
from evoagent.graders.multi_judge import MultiJudgeGrader

__all__ = ["EfficiencyGrader", "MultiJudgeGrader"]
```

- [ ] **Step 8: Commit**

```bash
git add evoagent/graders/ tests/unit/test_efficiency_grader.py tests/integration/test_multi_judge_grader.py
git commit -m "feat(evoagent): add EfficiencyGrader and MultiJudgeGrader"
```

---

## Task 8: Harness middleware

**Files:**
- Create: `evoagent/harness/middleware.py`
- Create: `evoagent/harness/builder.py`
- Create: `tests/integration/test_middleware.py`

- [ ] **Step 1: Write failing tests for middleware**

Create `tests/integration/test_middleware.py`:

```python
"""Tests for harness middleware."""

import json
import re

from evoagent.harness.middleware import (
    SelfVerificationMiddleware,
    LoopDetectionMiddleware,
    check_output,
    is_similar_query,
)


def test_check_output_detects_error():
    issues = check_output("Error: API quota exceeded")
    assert any("error" in i.lower() for i in issues)


def test_check_output_accepts_good_report():
    report = (
        "# Report\n## Summary\nThis is a finding about the topic.\n"
        "## Sources\n- Source 1\n" + "x" * 500
    )
    issues = check_output(report, required_sections=["summary", "finding", "source"])
    assert issues == []


def test_check_output_detects_short():
    issues = check_output("Too short", min_length=500)
    assert any("short" in i for i in issues)


def test_check_output_custom_sections():
    issues = check_output(
        "Some content " * 100,
        required_sections=["recommendation", "risk"],
    )
    assert any("recommendation" in i for i in issues)


def test_is_similar_query():
    assert is_similar_query("quantum computing advances", "advances in quantum computing")
    assert not is_similar_query("quantum computing", "weather forecast today")


def test_loop_detection_tracks_queries():
    mw = LoopDetectionMiddleware(max_similar=2, max_total=5)
    assert mw.should_warn("quantum computing") is False
    assert mw.should_warn("quantum computing research") is False
    assert mw.should_warn("quantum computing advances") is True  # 3rd similar
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd evoagent && python -m pytest tests/integration/test_middleware.py -v
```

- [ ] **Step 3: Implement `evoagent/harness/middleware.py`**

Adapt from `src/agent/middleware.py`. Key change: constructor parameters for `required_sections`, `error_patterns`, `min_length`, `max_similar`, `max_total`. Expose `check_output`, `is_similar_query`, and `LoopDetectionMiddleware.should_warn` as testable functions.

```python
"""Harness middleware stack for LangChain-based agents.

Four middleware classes, all parameterized:
1. SelfVerificationMiddleware — catches incomplete outputs
2. ContextAssemblyMiddleware — enriches first model call
3. LoopDetectionMiddleware — detects repetitive searches
4. TraceCaptureMiddleware — records traces for offline analysis
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Self-Verification
# ---------------------------------------------------------------------------

_DEFAULT_ERROR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:API|quota|rate.?limit|timeout|429|503)\s*(?:error|exceeded|limit)", re.I),
    re.compile(r"TIMEOUT after \d+s", re.I),
    re.compile(r"(?:search|tavily|duckduckgo).*(?:fail|error|unavailable)", re.I),
    re.compile(r"^Error:", re.I),
    re.compile(r"HTTPError|ConnectionError|RequestException", re.I),
]

_DEFAULT_REQUIRED_SECTIONS = ["summary", "finding", "source"]


def check_output(
    content: str,
    required_sections: list[str] | None = None,
    error_patterns: list[re.Pattern[str]] | None = None,
    min_length: int = 500,
) -> list[str]:
    """Check agent output for structural issues. Returns list of issue descriptions."""
    issues: list[str] = []
    patterns = error_patterns or _DEFAULT_ERROR_PATTERNS
    sections = required_sections or _DEFAULT_REQUIRED_SECTIONS

    for pattern in patterns:
        if pattern.search(content) and len(content) < 1000:
            issues.append("output appears to be an error message, not a report")
            break

    if len(content) < min_length:
        issues.append(f"output is too short (< {min_length} chars)")

    content_lower = content.lower()
    missing = [s for s in sections if s not in content_lower]
    if missing:
        issues.append(f"missing sections containing: {', '.join(missing)}")

    return issues


class SelfVerificationMiddleware:
    """Checks agent output for completeness before finishing."""

    def __init__(
        self,
        required_sections: list[str] | None = None,
        error_patterns: list[re.Pattern[str]] | None = None,
        min_length: int = 500,
        max_retries: int = 2,
    ) -> None:
        self._sections = required_sections or _DEFAULT_REQUIRED_SECTIONS
        self._patterns = error_patterns or _DEFAULT_ERROR_PATTERNS
        self._min_length = min_length
        self._max_retries = max_retries
        self._retry_count = 0

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage):
            return None
        if getattr(last_msg, "tool_calls", None):
            return None

        content = getattr(last_msg, "content", "") or ""
        if not isinstance(content, str):
            return None

        issues = check_output(content, self._sections, self._patterns, self._min_length)
        if not issues:
            self._retry_count = 0
            return None

        if self._retry_count >= self._max_retries:
            self._retry_count = 0
            return None

        self._retry_count += 1
        revision = HumanMessage(
            content=f"SELF-CHECK FAILED: {'; '.join(issues)}. Revise your report."
        )
        return {"messages": [*messages, revision]}


# ---------------------------------------------------------------------------
# 2. Context Assembly
# ---------------------------------------------------------------------------

class ContextAssemblyMiddleware:
    """Enriches first model call with environment context."""

    def __init__(self, skills_dir: Path | None = None) -> None:
        self._skills_dir = skills_dir
        self._first_call = True

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if not self._first_call:
            return None
        self._first_call = False

        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        if not messages:
            return None

        task_text = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                task_text = getattr(msg, "content", "") or ""
                break

        context_parts: list[str] = []

        if self._skills_dir and Path(self._skills_dir).exists():
            try:
                from evoagent.skills.manager import SkillManager
                skills = SkillManager(self._skills_dir).discover()
                if skills:
                    lines = [f"- **{s['name']}**: {s['description']}" for s in skills.values()]
                    context_parts.append("## Available Skills\n" + "\n".join(lines))
            except Exception:
                pass

        if not context_parts:
            return None

        context_block = "\n\n## Environment Context\n" + "\n\n".join(context_parts)
        new_messages = list(messages)
        for i, msg in enumerate(new_messages):
            if isinstance(msg, SystemMessage):
                new_messages[i] = SystemMessage(content=(getattr(msg, "content", "") or "") + context_block)
                return {"messages": new_messages}

        new_messages.insert(0, SystemMessage(content=context_block))
        return {"messages": new_messages}


# ---------------------------------------------------------------------------
# 3. Loop Detection
# ---------------------------------------------------------------------------

def is_similar_query(q1: str, q2: str) -> bool:
    """Check if two queries are substantially similar (>60% word overlap)."""
    words1 = set(re.sub(r"[^\w\s]", "", q1.lower()).split())
    words2 = set(re.sub(r"[^\w\s]", "", q2.lower()).split())
    if not words1 or not words2:
        return False
    return len(words1 & words2) / min(len(words1), len(words2)) > 0.6


class LoopDetectionMiddleware:
    """Detects repetitive search queries and nudges toward synthesis."""

    def __init__(self, max_similar: int = 3, max_total: int = 12) -> None:
        self._max_similar = max_similar
        self._max_total = max_total
        self._queries: list[str] = []
        self._warned = False

    def should_warn(self, query: str) -> bool:
        """Track a query and return True if a warning should be issued."""
        similar_count = sum(1 for q in self._queries if is_similar_query(q, query))
        self._queries.append(query)
        return (similar_count >= self._max_similar or len(self._queries) >= self._max_total) and not self._warned


# ---------------------------------------------------------------------------
# 4. Trace Capture
# ---------------------------------------------------------------------------

class TraceCaptureMiddleware:
    """Records execution traces to disk for offline analysis."""

    MAX_TRACE_BYTES = 100_000

    def __init__(self, task: str = "", traces_dir: Path | str | None = None) -> None:
        self._task = task
        self._traces_dir = Path(traces_dir) if traces_dir else Path("traces")
        self._run_id = str(uuid.uuid4())
        self._steps: list[dict[str, Any]] = []
        self._start_time: float = 0.0

    @property
    def run_id(self) -> str:
        return self._run_id

    def start(self) -> None:
        self._start_time = time.monotonic()

    def record_step(self, step: dict[str, Any]) -> None:
        self._steps.append(step)

    def save(self) -> Path | None:
        duration = time.monotonic() - self._start_time if self._start_time else 0.0
        trace = {
            "run_id": self._run_id,
            "task": self._task,
            "duration_seconds": round(duration, 2),
            "step_count": len(self._steps),
            "steps": self._steps,
        }

        self._traces_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self._traces_dir / f"{self._run_id}.json"

        try:
            trace_json = json.dumps(trace, default=str)
            if len(trace_json) > self.MAX_TRACE_BYTES:
                trace["steps"] = [
                    *trace["steps"][:5],
                    {"type": "truncated", "removed_steps": len(trace["steps"]) - 10},
                    *trace["steps"][-5:],
                ]
                trace_json = json.dumps(trace, default=str)
            trace_path.write_text(trace_json)
            return trace_path
        except Exception as exc:
            logger.warning("Failed to save trace: %s", exc)
            return None
```

- [ ] **Step 4: Implement `evoagent/harness/builder.py`**

```python
"""Factory for building default middleware stacks."""

from __future__ import annotations

from pathlib import Path

from evoagent.harness.middleware import (
    ContextAssemblyMiddleware,
    LoopDetectionMiddleware,
    SelfVerificationMiddleware,
    TraceCaptureMiddleware,
)


def default_middleware_stack(
    task: str = "",
    skills_dir: Path | None = None,
    traces_dir: Path | None = None,
    required_sections: list[str] | None = None,
) -> list:
    """Build the default harness middleware stack.

    Returns a list of middleware instances in recommended order.
    """
    return [
        SelfVerificationMiddleware(required_sections=required_sections),
        ContextAssemblyMiddleware(skills_dir=skills_dir),
        LoopDetectionMiddleware(),
        TraceCaptureMiddleware(task=task, traces_dir=traces_dir),
    ]
```

- [ ] **Step 5: Run middleware tests**

```bash
cd evoagent && python -m pytest tests/integration/test_middleware.py -v
```

Expected: all PASS.

- [ ] **Step 6: Update `evoagent/harness/__init__.py`**

```python
"""Harness middleware stack for LangChain agents."""

try:
    from langchain_core.messages import BaseMessage as _  # noqa: F401
except ImportError:
    raise ImportError(
        "evoagent.harness requires langchain-core. "
        "Install with: pip install evoagent[harness]"
    ) from None

from evoagent.harness.builder import default_middleware_stack
from evoagent.harness.middleware import (
    ContextAssemblyMiddleware,
    LoopDetectionMiddleware,
    SelfVerificationMiddleware,
    TraceCaptureMiddleware,
)

__all__ = [
    "ContextAssemblyMiddleware",
    "LoopDetectionMiddleware",
    "SelfVerificationMiddleware",
    "TraceCaptureMiddleware",
    "default_middleware_stack",
]
```

- [ ] **Step 7: Commit**

```bash
git add evoagent/harness/ tests/integration/test_middleware.py
git commit -m "feat(evoagent): add harness middleware with parameterized SelfVerification, LoopDetection, ContextAssembly, TraceCapture"
```

---

## Task 9: Tracing module

**Files:**
- Create: `evoagent/tracing/trajectory.py`
- Create: `evoagent/tracing/capture.py`

- [ ] **Step 1: Implement `evoagent/tracing/trajectory.py`**

Copy from `src/tracing/trajectory.py` with import path fix:

```python
"""Trajectory data models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool invocation within a trajectory."""

    name: str
    args: dict = Field(default_factory=dict)
    output: str = ""


class TrajectoryRecord(BaseModel):
    """A complete agent run trajectory."""

    run_id: str
    task: str = ""
    output: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    total_tokens: int = 0
    total_steps: int = 0
    latency_seconds: float = 0.0
    tool_call_count: int = 0
    status: str = "completed"

    @property
    def is_successful(self) -> bool:
        return self.status == "completed" and bool(self.output)
```

- [ ] **Step 2: Implement `evoagent/tracing/capture.py`**

Re-export `TraceCaptureMiddleware` from harness for convenience:

```python
"""Trace capture re-exports for convenience."""

from evoagent.harness.middleware import TraceCaptureMiddleware

__all__ = ["TraceCaptureMiddleware"]
```

- [ ] **Step 3: Update `evoagent/tracing/__init__.py`**

```python
"""Trace capture and trajectory models."""

from evoagent.tracing.trajectory import TrajectoryRecord, ToolCall

__all__ = ["TrajectoryRecord", "ToolCall"]
```

- [ ] **Step 4: Commit**

```bash
git add evoagent/tracing/
git commit -m "feat(evoagent): add tracing module with TrajectoryRecord model"
```

---

## Task 10: Evolution loop and analyzer

**Files:**
- Create: `evoagent/evolution/analyzer.py`
- Create: `evoagent/evolution/loop.py`
- Create: `evoagent/evolution/sleep_review.py`
- Create: `evoagent/evolution/prompt_optimizer.py`
- Create: `evoagent/evolution/state.py`
- Create: `tests/integration/test_analyzer.py`
- Create: `tests/e2e/test_evolution_loop.py`

- [ ] **Step 1: Write failing tests for analyzer**

Create `tests/integration/test_analyzer.py`:

```python
"""Tests for trajectory analyzer."""

from evoagent.core.types import GraderResult, TrajectoryMetrics
from evoagent.evolution.analyzer import classify_trajectory, analyze_trajectory
from evoagent.graders.efficiency import EfficiencyGrader


def test_classify_successful():
    results = [
        GraderResult(name="a", score=0.9, passed=True, reasoning="good"),
        GraderResult(name="b", score=0.8, passed=True, reasoning="good"),
        GraderResult(name="c", score=0.7, passed=False, reasoning="ok"),
    ]
    classification, avg = classify_trajectory(results)
    assert classification == "successful"
    assert avg > 0.75


def test_classify_failed():
    results = [
        GraderResult(name="a", score=0.2, passed=False, reasoning="bad"),
        GraderResult(name="b", score=0.3, passed=False, reasoning="bad"),
    ]
    classification, avg = classify_trajectory(results)
    assert classification == "failed"


def test_classify_empty():
    classification, avg = classify_trajectory([])
    assert classification == "failed"
    assert avg == 0.0


def test_analyze_with_graders():
    graders = [EfficiencyGrader()]
    results = analyze_trajectory(
        graders=graders,
        task="test task",
        output="test output",
        metrics=TrajectoryMetrics(total_tokens=5000, total_steps=3, latency_seconds=15.0),
    )
    assert len(results) == 1
    assert results[0].name == "efficiency"
```

- [ ] **Step 2: Implement `evoagent/evolution/analyzer.py`**

```python
"""Trajectory analysis: run graders and classify results.

Accepts any list of Grader implementations. No hardcoded graders.
"""

from __future__ import annotations

import logging
from typing import Any

from evoagent.core.protocols import Grader
from evoagent.core.types import GraderResult, TrajectoryMetrics

logger = logging.getLogger(__name__)

SUCCESSFUL_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.50
MIN_PASS_COUNT = 2
MIN_AVERAGE_SCORE = 0.60


def classify_trajectory(grader_results: list[GraderResult]) -> tuple[str, float]:
    """Classify a trajectory based on grader results.

    Returns (classification, average_score) where classification is
    one of "successful", "partial", "failed".
    """
    if not grader_results:
        return "failed", 0.0

    scores = [g.score for g in grader_results]
    avg_score = sum(scores) / len(scores)
    pass_count = sum(1 for g in grader_results if g.passed)

    if pass_count >= MIN_PASS_COUNT and avg_score >= MIN_AVERAGE_SCORE:
        if avg_score >= SUCCESSFUL_THRESHOLD:
            return "successful", round(avg_score, 3)
        return "partial", round(avg_score, 3)

    if avg_score >= PARTIAL_THRESHOLD:
        return "partial", round(avg_score, 3)

    return "failed", round(avg_score, 3)


def analyze_trajectory(
    graders: list[Grader],
    task: str,
    output: str,
    metrics: TrajectoryMetrics | None = None,
    **kwargs: Any,
) -> list[GraderResult]:
    """Run all graders on a (task, output) pair and return results.

    Args:
        graders: List of Grader implementations to run.
        task: The original task description.
        output: The agent's output text.
        metrics: Optional efficiency metrics.
        **kwargs: Extra context passed to each grader.

    Returns:
        List of GraderResult, one per grader.
    """
    if metrics is not None:
        kwargs["metrics"] = metrics

    results: list[GraderResult] = []
    for grader in graders:
        try:
            result = grader.grade(task=task, output=output, **kwargs)
            results.append(result)
        except Exception as exc:
            logger.error("Grader '%s' failed: %s", grader.name, exc)
            results.append(GraderResult(
                name=grader.name, score=0.0, passed=False,
                reasoning=f"grader_error: {exc}",
            ))
    return results
```

- [ ] **Step 3: Run analyzer tests**

```bash
cd evoagent && python -m pytest tests/integration/test_analyzer.py -v
```

Expected: all PASS.

- [ ] **Step 4: Implement `evoagent/evolution/state.py`**

```python
"""Evolution state persistence.

Writes cycle metrics, analyses, and plateau reports to disk
for the outer-loop coding agent to read.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def persist_evolution_state(
    state_dir: Path,
    cycle: int,
    metrics: dict[str, Any],
    analyses: list[dict[str, Any]],
    plateau_reason: str | None = None,
) -> None:
    """Write evolution state to disk."""
    state_dir.mkdir(parents=True, exist_ok=True)

    (state_dir / f"cycle_{cycle}_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str)
    )
    (state_dir / f"cycle_{cycle}_analyses.json").write_text(
        json.dumps(analyses, indent=2, default=str)
    )

    if plateau_reason:
        (state_dir / "plateau_report.md").write_text(
            f"# Plateau Report\n\nCycle: {cycle}\nReason: {plateau_reason}\n"
        )

    logger.info("Persisted evolution state for cycle %d", cycle)
```

- [ ] **Step 5: Implement `evoagent/evolution/sleep_review.py`**

Adapt from `src/agent/sleep_review.py`. Accept `MemoryBackend` protocol instead of concrete store.

```python
"""Sleep-time compute: offline cross-run trace analysis.

Reviews execution traces across multiple runs to discover cross-cutting
patterns and generate meta-instructions stored as semantic memories.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evoagent.core.parsing import parse_llm_json
from evoagent.core.protocols import MemoryBackend

logger = logging.getLogger(__name__)

SLEEP_REVIEW_PROMPT = (
    "You are analyzing execution traces from multiple agent runs.\n\n"
    "## Episodic Memories\n{episodic_summaries}\n\n"
    "## Execution Traces\n{trace_summaries}\n\n"
    "Identify:\n"
    "1. consistent_successes\n2. recurring_failures\n"
    "3. task_type_insights\n4. meta_instructions (3-5 rules)\n\n"
    "Respond as JSON with those 4 keys (each a list of strings)."
)


def run_sleep_review(
    llm: BaseChatModel,
    memory: MemoryBackend,
    traces_dir: Path,
    prompt_template: str | None = None,
    max_traces: int = 20,
) -> dict[str, Any]:
    """Run sleep-time review and store meta-instructions."""
    traces = _load_traces(traces_dir, limit=max_traces)
    if not traces:
        return {"meta_instructions": [], "status": "no_traces"}

    trace_summaries = "\n".join(f"- {_summarize_trace(t)}" for t in traces)

    episodic_items = memory.list_all("episodic")
    ep_lines = []
    for _key, data in list(episodic_items.items())[:15]:
        ep_lines.append(
            f"- Task: {data.get('task', '?')[:80]} | Score: {data.get('score', '?')}"
        )
    episodic_summaries = "\n".join(ep_lines) or "No episodic memories yet."

    template = prompt_template or SLEEP_REVIEW_PROMPT
    prompt = template.format(
        episodic_summaries=episodic_summaries,
        trace_summaries=trace_summaries,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    result = parse_llm_json(response.content)

    stored = 0
    for instruction in result.get("meta_instructions", []):
        if isinstance(instruction, str) and instruction.strip():
            memory.store("semantic", f"meta-{uuid.uuid4().hex[:8]}", {
                "type": "meta_instruction",
                "content": instruction.strip(),
                "source": "sleep_review",
            })
            stored += 1

    logger.info("Sleep review: %d traces, %d meta-instructions stored", len(traces), stored)

    return {
        "meta_instructions": result.get("meta_instructions", []),
        "consistent_successes": result.get("consistent_successes", []),
        "recurring_failures": result.get("recurring_failures", []),
        "status": "complete",
    }


def _load_traces(traces_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    if not traces_dir.exists():
        return []
    files = sorted(traces_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    traces = []
    for f in files[:limit]:
        try:
            traces.append(json.loads(f.read_text()))
        except Exception:
            continue
    return traces


def _summarize_trace(trace: dict[str, Any]) -> str:
    task = trace.get("task", "unknown")[:100]
    duration = trace.get("duration_seconds", "?")
    steps = trace.get("step_count", 0)
    return f"Task: {task} | Duration: {duration}s, Steps: {steps}"
```

- [ ] **Step 6: Implement `evoagent/evolution/prompt_optimizer.py`**

```python
"""Metaprompt-based prompt optimization.

Analyzes failures, builds a metaprompt, and generates an improved
system prompt. Includes autonomy validation to prevent prompt drift.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evoagent.core.protocols import PromptStore

logger = logging.getLogger(__name__)

MAX_AUTONOMY_RETRIES = 2

_AUTONOMY_VIOLATIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ask\s+(the\s+)?user", re.I), "asks the user"),
    (re.compile(r"would you like", re.I), "asks 'would you like'"),
    (re.compile(r"do you (?:want|prefer|need)", re.I), "asks user preference"),
    (re.compile(r"let me know", re.I), "requests user feedback"),
]

DEFAULT_METAPROMPT = (
    "You are a prompt engineering expert.\n\n"
    "## Current Prompt\n{current_prompt}\n\n"
    "## Current Score\n{current_score}\n\n"
    "## Failure Analysis\n{failure_analysis}\n\n"
    "## Common Issues\n{common_issues}\n\n"
    "Generate an improved system prompt that addresses the failures "
    "while preserving what works. Output ONLY the new prompt text."
)


def validate_prompt_autonomy(prompt: str) -> list[str]:
    """Check for patterns that violate autonomous operation."""
    return [desc for pattern, desc in _AUTONOMY_VIOLATIONS if pattern.search(prompt)]


def optimize_prompt(
    llm: BaseChatModel,
    prompt_store: PromptStore,
    analyses: list[dict[str, Any]],
    metaprompt_template: str | None = None,
) -> int:
    """Generate an improved prompt based on failure analysis.

    Returns the new prompt version number.
    """
    version, current_prompt = prompt_store.get_current()

    failed = [a for a in analyses if a.get("classification") in ("failed", "partial")]
    if not failed:
        return version

    issues: list[str] = []
    for a in failed:
        for g in a.get("grader_results", []):
            if isinstance(g, dict) and not g.get("passed", True):
                issues.append(f"[{g.get('name', '?')}] {g.get('reasoning', '')[:150]}")

    template = metaprompt_template or DEFAULT_METAPROMPT
    scores = [a.get("average_score", 0) for a in analyses]
    avg_score = sum(scores) / len(scores) if scores else 0

    metaprompt = template.format(
        current_prompt=current_prompt[:3000],
        current_score=f"{avg_score:.3f}",
        failure_analysis=f"{len(failed)} failed/partial out of {len(analyses)}",
        common_issues="\n".join(f"- {i}" for i in issues[:10]),
    )

    for attempt in range(1 + MAX_AUTONOMY_RETRIES):
        response = llm.invoke([HumanMessage(content=metaprompt)])
        new_prompt = response.content.strip()

        violations = validate_prompt_autonomy(new_prompt)
        if not violations:
            return prompt_store.save(new_prompt, score=None, parent=version)

        logger.warning("Autonomy violations (attempt %d): %s", attempt + 1, violations)
        metaprompt += f"\n\nThe previous attempt had violations: {violations}. Fix them."

    logger.error("Failed to generate autonomous prompt after %d attempts", MAX_AUTONOMY_RETRIES + 1)
    return version
```

- [ ] **Step 7: Write E2E test for evolution loop**

Create `tests/e2e/test_evolution_loop.py`:

```python
"""E2E test for the full evolution loop with a toy agent."""

from evoagent.core.config import EvoAgentConfig
from evoagent.core.protocols import AgentFactory
from evoagent.core.types import TaskResult, TrajectoryMetrics
from evoagent.graders.efficiency import EfficiencyGrader
from evoagent.evolution.analyzer import analyze_trajectory, classify_trajectory
from evoagent.memory.store import FileMemoryStore
from evoagent.skills.manager import SkillManager


class ToyAgent(AgentFactory):
    """Returns canned responses for testing."""

    def create(self, system_prompt, middleware, **kwargs):
        return system_prompt  # agent is just the prompt string

    def run(self, agent, task, timeout=60):
        return TaskResult(
            task=task,
            output=f"# Report\n## Summary\nAnalysis of {task}.\n## Finding\nKey finding.\n## Source\n- Source 1\n" + "x" * 500,
            tool_calls=[],
            duration_seconds=10.0,
            status="success",
        )


def test_full_grading_cycle(tmp_path):
    """Test: create agent -> run -> grade -> classify."""
    factory = ToyAgent()
    agent = factory.create(system_prompt="You are helpful.", middleware=[])
    result = factory.run(agent, task="Test quantum computing")

    graders = [EfficiencyGrader()]
    grades = analyze_trajectory(
        graders=graders,
        task=result.task,
        output=result.output,
        metrics=TrajectoryMetrics(total_tokens=5000, total_steps=3, latency_seconds=10.0),
    )
    classification, avg_score = classify_trajectory(grades)
    assert classification in ("successful", "partial", "failed")
    assert 0 <= avg_score <= 1


def test_memory_persists_across_cycles(tmp_path):
    """Test: memory store works across simulated cycles."""
    store = FileMemoryStore(tmp_path / "memory")
    store.store("episodic", "cycle-1", {"task": "test", "score": 0.7})
    store.store("episodic", "cycle-2", {"task": "test", "score": 0.85})

    all_ep = store.list_all("episodic")
    assert len(all_ep) == 2


def test_skills_created_during_cycle(tmp_path):
    """Test: skill store works during simulated cycle."""
    mgr = SkillManager(tmp_path / "skills")
    mgr.create("test-skill", "A test skill", "# Do the thing")
    assert "test-skill" in mgr.discover()
```

- [ ] **Step 8: Run all tests**

```bash
cd evoagent && python -m pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 9: Update `evoagent/evolution/__init__.py`**

```python
"""Evolution loop: propose -> run -> grade -> accept/reject."""

try:
    from langgraph.graph import StateGraph as _  # noqa: F401
except ImportError:
    raise ImportError(
        "evoagent.evolution requires langgraph. "
        "Install with: pip install evoagent[evolution]"
    ) from None

from evoagent.evolution.analyzer import analyze_trajectory, classify_trajectory
from evoagent.evolution.sleep_review import run_sleep_review

__all__ = ["analyze_trajectory", "classify_trajectory", "run_sleep_review"]
```

- [ ] **Step 10: Commit**

```bash
git add evoagent/evolution/ evoagent/tracing/ tests/integration/test_analyzer.py tests/e2e/test_evolution_loop.py
git commit -m "feat(evoagent): add evolution loop, analyzer, prompt optimizer, sleep review, and tracing"
```

---

## Task 11: Research agent example

**Files:**
- Create: `examples/research_agent/README.md`
- Create: `examples/research_agent/main.py`

- [ ] **Step 1: Create `examples/research_agent/README.md`**

```markdown
# Self-Improving Research Agent

Example application using the `evoagent` library to build a research agent
that improves itself through the Karpathy-style evolution loop.

## Install

```bash
pip install evoagent[all]
pip install deepagents langchain-openai langchain-tavily ddgs
```

## Run

```bash
# Single task
python main.py run "What are the latest advances in quantum computing?"

# Evolution loop (3 cycles)
python main.py evolve --tasks tasks.json --cycles 3

# Sleep-time review
python main.py sleep-review
```
```

- [ ] **Step 2: Create `examples/research_agent/main.py`**

```python
"""Research agent using evoagent library — demonstrates full Karpathy loop."""

import argparse
import json

from evoagent import EvoAgentConfig
from evoagent.graders import EfficiencyGrader, MultiJudgeGrader
from evoagent.harness import default_middleware_stack
from evoagent.memory import FileMemoryStore
from evoagent.skills import SkillManager
from evoagent.evolution import analyze_trajectory, classify_trajectory


def main():
    parser = argparse.ArgumentParser(description="Self-improving research agent")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run").add_argument("task")
    evolve = sub.add_parser("evolve")
    evolve.add_argument("--tasks", required=True)
    evolve.add_argument("--cycles", type=int, default=3)
    sub.add_parser("sleep-review")

    args = parser.parse_args()
    config = EvoAgentConfig(base_dir="./data")

    if args.command == "run":
        print(f"Running task: {args.task}")
        # Users would implement their AgentFactory here
        print("See README.md for full AgentFactory implementation example.")

    elif args.command == "evolve":
        tasks = json.loads(open(args.tasks).read())
        print(f"Running evolution: {len(tasks)} tasks, {args.cycles} cycles")

    elif args.command == "sleep-review":
        print("Running sleep-time review...")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add examples/
git commit -m "docs(evoagent): add research agent example"
```

---

## Task 12: Final integration test and cleanup

**Files:**
- Modify: `evoagent/__init__.py` (add evolution exports)

- [ ] **Step 1: Run full test suite**

```bash
cd evoagent && python -m pytest tests/ -v --tb=short
```

Expected: all PASS.

- [ ] **Step 2: Run linter**

```bash
cd evoagent && ruff check evoagent/ tests/
```

Fix any issues.

- [ ] **Step 3: Verify import tiers work**

```python
# Tier 1: core only (no langchain needed)
python -c "from evoagent.core.types import GraderResult; from evoagent.memory import FileMemoryStore; from evoagent.skills import SkillManager; print('Tier 1 OK')"

# Tier 2: harness + graders (needs langchain)
python -c "from evoagent.harness import default_middleware_stack; from evoagent.graders import EfficiencyGrader, MultiJudgeGrader; print('Tier 2 OK')"

# Tier 3: evolution (needs langgraph)
python -c "from evoagent.evolution import analyze_trajectory, classify_trajectory, run_sleep_review; print('Tier 3 OK')"
```

- [ ] **Step 4: Commit final cleanup**

```bash
git add -A
git commit -m "chore(evoagent): final cleanup, all tests passing"
```
