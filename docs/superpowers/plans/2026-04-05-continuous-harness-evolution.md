# Continuous Harness & Context Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the agent self-improve its harness (middleware parameters, tool configs, completion logic) and context (memory, prompts, skills) continuously from every single user interaction — no explicit `evolve` call needed.

**Architecture:** A `HarnessConfig` stores all tunable middleware parameters as a versioned JSON file. Every single run captures full execution traces and harness performance signals. The daemon's `_run_evolution_cycle` is extended to optimize harness config alongside prompts, using raw trace data (not summaries) for counterfactual diagnosis. Environment bootstrapping and time budget awareness are wired into every run.

**Tech Stack:** Python 3.12+, pydantic, langgraph, langchain-core, existing evoagent library

---

### Task 1: HarnessConfig — Versioned Tunable Parameters

**Files:**
- Create: `src/evolution/harness_config.py`
- Test: `tests/unit/test_harness_config.py`

- [ ] **Step 1: Write the failing test for HarnessConfig load/save**

```python
# tests/unit/test_harness_config.py
"""Tests for harness config versioning."""
import json
from pathlib import Path

from src.evolution.harness_config import HarnessConfig, HarnessConfigStore


class TestHarnessConfig:
    def test_defaults(self):
        cfg = HarnessConfig()
        assert cfg.self_verification_max_retries == 2
        assert cfg.self_verification_min_length == 500
        assert cfg.loop_detection_max_similar == 3
        assert cfg.loop_detection_max_total == 12
        assert cfg.loop_detection_max_file_edits == 5
        assert cfg.time_budget_seconds == 300
        assert cfg.time_budget_warn_at == [0.6, 0.85]
        assert cfg.reasoning_planning_effort == "high"
        assert cfg.reasoning_implementation_effort == "medium"
        assert cfg.reasoning_verification_effort == "high"
        assert cfg.context_assembly_detect_env is True
        assert cfg.self_verification_verify_against_task is False

    def test_to_dict_and_from_dict(self):
        cfg = HarnessConfig(self_verification_max_retries=5, time_budget_seconds=180)
        d = cfg.to_dict()
        restored = HarnessConfig.from_dict(d)
        assert restored.self_verification_max_retries == 5
        assert restored.time_budget_seconds == 180


class TestHarnessConfigStore:
    def test_save_and_load(self, tmp_path):
        store = HarnessConfigStore(tmp_path / "harness")
        cfg = HarnessConfig(loop_detection_max_similar=5)
        store.save(cfg, score=0.82)
        loaded = store.load_best()
        assert loaded.loop_detection_max_similar == 5

    def test_load_best_returns_highest_score(self, tmp_path):
        store = HarnessConfigStore(tmp_path / "harness")
        store.save(HarnessConfig(time_budget_seconds=300), score=0.70)
        store.save(HarnessConfig(time_budget_seconds=180), score=0.85)
        store.save(HarnessConfig(time_budget_seconds=120), score=0.60)
        best = store.load_best()
        assert best.time_budget_seconds == 180

    def test_load_best_returns_latest_when_unscored(self, tmp_path):
        store = HarnessConfigStore(tmp_path / "harness")
        store.save(HarnessConfig(time_budget_seconds=300))
        store.save(HarnessConfig(time_budget_seconds=180))
        best = store.load_best()
        assert best.time_budget_seconds == 180

    def test_update_score_incremental(self, tmp_path):
        store = HarnessConfigStore(tmp_path / "harness")
        store.save(HarnessConfig(), score=0.80)
        store.update_score(1, 0.60)
        # Incremental avg: (0.80 + 0.60) / 2 = 0.70
        data = json.loads((tmp_path / "harness" / "v0001.json").read_text())
        assert abs(data["score"] - 0.70) < 0.01
        assert data["score_count"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_harness_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evolution.harness_config'`

- [ ] **Step 3: Implement HarnessConfig and HarnessConfigStore**

```python
# src/evolution/harness_config.py
"""Versioned harness configuration.

Stores all tunable middleware parameters as a versioned JSON file.
The daemon evolves these alongside prompts based on run performance signals.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HarnessConfig:
    """All tunable middleware parameters in one place."""

    # SelfVerificationMiddleware
    self_verification_max_retries: int = 2
    self_verification_min_length: int = 500
    self_verification_verify_against_task: bool = False
    self_verification_required_sections: list[str] = field(
        default_factory=lambda: ["summary", "finding", "source"]
    )

    # LoopDetectionMiddleware
    loop_detection_max_similar: int = 3
    loop_detection_max_total: int = 12
    loop_detection_max_file_edits: int = 5
    loop_detection_max_repeated_tools: int = 4

    # TimeBudgetMiddleware
    time_budget_seconds: int = 300
    time_budget_warn_at: list[float] = field(default_factory=lambda: [0.6, 0.85])

    # ReasoningSandwichMiddleware
    reasoning_planning_effort: str = "high"
    reasoning_implementation_effort: str = "medium"
    reasoning_verification_effort: str = "high"
    reasoning_planning_calls: int = 2

    # ContextAssemblyMiddleware
    context_assembly_detect_env: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarnessConfig":
        # Filter to only known fields
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in valid})


class HarnessConfigStore:
    """File-based versioned harness config store."""

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _version_path(self, version: int) -> Path:
        return self.config_dir / f"v{version:04d}.json"

    def _all_versions(self) -> list[dict[str, Any]]:
        versions = []
        for path in sorted(self.config_dir.glob("v*.json")):
            with open(path) as f:
                versions.append(json.load(f))
        return versions

    def save(self, config: HarnessConfig, score: float | None = None) -> int:
        versions = self._all_versions()
        version = len(versions) + 1
        data = {
            "version": version,
            "config": config.to_dict(),
            "score": score,
            "score_count": 1 if score is not None else 0,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        path = self._version_path(version)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved harness config v%d (score=%s)", version, score)
        return version

    def load_best(self) -> HarnessConfig:
        versions = self._all_versions()
        if not versions:
            return HarnessConfig()
        scored = [v for v in versions if v.get("score") is not None]
        if scored:
            best = max(scored, key=lambda v: v["score"])
        else:
            best = versions[-1]
        return HarnessConfig.from_dict(best["config"])

    def get_latest_version(self) -> int:
        versions = self._all_versions()
        return len(versions)

    def update_score(self, version: int, score: float) -> None:
        path = self._version_path(version)
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        n = data.get("score_count", 0)
        old_score = data.get("score")
        if n == 0 or old_score is None:
            data["score"] = score
            data["score_count"] = 1
        else:
            n += 1
            data["score"] = old_score + (score - old_score) / n
            data["score_count"] = n
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_harness_config.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/evolution/harness_config.py tests/unit/test_harness_config.py
git commit -m "feat: add HarnessConfig versioned tunable parameters store"
```

---

### Task 2: Wire HarnessConfig Into Agent Creation

**Files:**
- Modify: `src/agent/deep_agent.py:113-177` (the `create_agent` function)
- Modify: `src/config/settings.py` (add `harness_config_dir` setting)
- Modify: `evoagent/harness/builder.py:18-48` (accept HarnessConfig)
- Test: `tests/unit/test_harness_config.py` (extend with integration test)

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_harness_config.py

class TestHarnessConfigWiring:
    def test_build_middleware_from_config(self):
        from evoagent.harness.builder import default_middleware_stack
        from src.evolution.harness_config import HarnessConfig

        cfg = HarnessConfig(
            time_budget_seconds=120,
            loop_detection_max_similar=7,
            self_verification_max_retries=5,
            context_assembly_detect_env=True,
        )
        stack = default_middleware_stack(harness_config=cfg)

        # Find each middleware by type name
        names = [type(m).__name__ for m in stack]
        assert "TimeBudgetMiddleware" in names
        assert "LoopDetectionMiddleware" in names
        assert "SelfVerificationMiddleware" in names
        assert "ContextAssemblyMiddleware" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_harness_config.py::TestHarnessConfigWiring -v`
Expected: FAIL — `default_middleware_stack() got an unexpected keyword argument 'harness_config'`

- [ ] **Step 3: Update `default_middleware_stack` to accept HarnessConfig**

In `evoagent/harness/builder.py`, update the function signature and body:

```python
def default_middleware_stack(
    task: str = "",
    skills_dir: Path | None = None,
    traces_dir: Path | None = None,
    required_sections: list[str] | None = None,
    detect_env: bool = False,
    working_dir: Path | None = None,
    budget_seconds: int = 300,
    verify_against_task: bool = False,
    llm=None,
    harness_config=None,
) -> list:
    """Build the default middleware stack.

    If harness_config is provided, its values override the individual parameters.
    """
    if harness_config is not None:
        required_sections = required_sections or harness_config.self_verification_required_sections
        detect_env = harness_config.context_assembly_detect_env
        budget_seconds = harness_config.time_budget_seconds
        verify_against_task = harness_config.self_verification_verify_against_task

    return [
        SelfVerificationMiddleware(
            required_sections=required_sections,
            max_retries=harness_config.self_verification_max_retries if harness_config else 2,
            min_length=harness_config.self_verification_min_length if harness_config else 500,
            verify_against_task=verify_against_task,
            llm=llm,
        ),
        ContextAssemblyMiddleware(
            skills_dir=skills_dir,
            detect_env=detect_env,
            working_dir=working_dir,
        ),
        LoopDetectionMiddleware(
            max_similar=harness_config.loop_detection_max_similar if harness_config else 3,
            max_total=harness_config.loop_detection_max_total if harness_config else 12,
            max_file_edits=harness_config.loop_detection_max_file_edits if harness_config else 5,
            max_repeated_tools=harness_config.loop_detection_max_repeated_tools if harness_config else 4,
        ),
        TimeBudgetMiddleware(
            budget_seconds=budget_seconds,
            warn_at=harness_config.time_budget_warn_at if harness_config else [0.6, 0.85],
        ),
        ReasoningSandwichMiddleware(
            planning_effort=harness_config.reasoning_planning_effort if harness_config else "high",
            implementation_effort=harness_config.reasoning_implementation_effort if harness_config else "medium",
            verification_effort=harness_config.reasoning_verification_effort if harness_config else "high",
            planning_calls=harness_config.reasoning_planning_calls if harness_config else 2,
        ),
        TraceCaptureMiddleware(task=task, traces_dir=traces_dir),
    ]
```

- [ ] **Step 4: Add harness_config_dir to Settings**

In `src/config/settings.py`, add after `evolution_state_dir`:

```python
    harness_config_dir: str = "harness_config"
```

And add the property:

```python
    @property
    def harness_config_path(self) -> Path:
        return PROJECT_ROOT / self.harness_config_dir
```

- [ ] **Step 5: Wire HarnessConfig into `create_agent`**

In `src/agent/deep_agent.py`, update `create_agent` to load harness config:

```python
def create_agent(
    settings: Settings,
    prompt_store: PromptStore,
    memory_store: FileMemoryStore,
    task: str = "",
    extra_tools: list[BaseTool] | None = None,
) -> CompiledStateGraph[Any, Any]:
    """Create the deep research agent with memory-augmented prompts."""
    from src.evolution.harness_config import HarnessConfig, HarnessConfigStore

    llm = create_llm(settings)
    search_tool = create_search_tool(settings)
    tools: list[BaseTool] = [search_tool]
    if extra_tools:
        tools.extend(extra_tools)

    prompt = prompt_store.get_current_prompt() or DEFAULT_SYSTEM_PROMPT
    memory_context = (
        build_memory_context(memory_store, task, token_budget=settings.memory_token_budget)
        if task
        else ""
    )
    prompt = prompt.format(memory_context=memory_context)

    skills_sources: list[str] = []
    if settings.skills_path.exists():
        skills_sources = [str(settings.skills_path)]

    subagents = None
    if settings.use_subagents:
        from src.agent.subagents import build_research_subagent, build_synthesis_subagent
        subagents = [
            build_research_subagent(settings),
            build_synthesis_subagent(settings),
        ]

    # Load best harness config (or defaults)
    harness_store = HarnessConfigStore(settings.harness_config_path)
    harness_cfg = harness_store.load_best()

    logger.info(
        "Creating agent '%s' with model=%s, tools=%d, skills=%d%s",
        AGENT_NAME, settings.model, len(tools), len(skills_sources),
        f", subagents={len(subagents)}" if subagents else "",
    )

    harness_middleware = default_middleware_stack(
        task=task,
        skills_dir=settings.skills_path,
        traces_dir=settings.traces_path,
        harness_config=harness_cfg,
        llm=llm,
    )

    kwargs: dict[str, Any] = {
        "model": llm,
        "tools": tools,
        "system_prompt": prompt,
        "name": AGENT_NAME,
        "skills": skills_sources or None,
        "middleware": harness_middleware,
    }
    if subagents:
        kwargs["subagents"] = subagents

    return create_deep_agent(**kwargs)
```

- [ ] **Step 6: Run all tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_harness_config.py tests/evoagent/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/evolution/harness_config.py src/agent/deep_agent.py src/config/settings.py evoagent/harness/builder.py tests/unit/test_harness_config.py
git commit -m "feat: wire HarnessConfig into agent creation and middleware stack"
```

---

### Task 3: Capture Full Execution Traces in RunLog

**Files:**
- Modify: `src/evolution/run_log.py` (add trace_path field to RunLogEntry)
- Modify: `src/cli/commands.py:24-146` (capture trace path from TraceCaptureMiddleware and pass to run log)
- Test: `tests/unit/test_harness_config.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_harness_config.py

class TestRunLogTraceCapture:
    def test_run_log_entry_has_trace_path(self):
        from src.evolution.run_log import RunLogEntry
        entry = RunLogEntry(
            run_id="test-001",
            task="test task",
            output="report",
            classification="partial",
            average_score=0.78,
            grader_results=[],
            prompt_version=7,
            harness_config_version=1,
            trace_path="/tmp/traces/test.json",
        )
        d = entry.to_dict()
        assert d["trace_path"] == "/tmp/traces/test.json"
        assert d["harness_config_version"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_harness_config.py::TestRunLogTraceCapture -v`
Expected: FAIL — `RunLogEntry.__init__() got an unexpected keyword argument 'harness_config_version'`

- [ ] **Step 3: Extend RunLogEntry with trace_path and harness_config_version**

In `src/evolution/run_log.py`, update the `RunLogEntry.__init__` and `to_dict`:

```python
class RunLogEntry:
    """A single run result stored in the log."""

    def __init__(
        self,
        run_id: str,
        task: str,
        output: str,
        classification: str,
        average_score: float,
        grader_results: list[dict[str, Any]],
        prompt_version: int,
        timestamp: str | None = None,
        processed: bool = False,
        harness_config_version: int = 0,
        trace_path: str = "",
    ) -> None:
        self.run_id = run_id
        self.task = task
        self.output = output
        self.classification = classification
        self.average_score = average_score
        self.grader_results = grader_results
        self.prompt_version = prompt_version
        self.timestamp = timestamp or datetime.now(UTC).isoformat()
        self.processed = processed
        self.harness_config_version = harness_config_version
        self.trace_path = trace_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "output": self.output,
            "classification": self.classification,
            "average_score": self.average_score,
            "grader_results": self.grader_results,
            "prompt_version": self.prompt_version,
            "timestamp": self.timestamp,
            "processed": self.processed,
            "harness_config_version": self.harness_config_version,
            "trace_path": self.trace_path,
        }
```

- [ ] **Step 4: Update `cmd_run` to capture trace path and harness config version**

In `src/cli/commands.py`, update the `cmd_run` function. After the agent runs and before grading, extract the trace path from the middleware. Then update the RunLogEntry creation to include both new fields. The key changes are:

1. After `create_agent()`, extract the TraceCaptureMiddleware from the harness and call `.save()` to persist the trace
2. Load the HarnessConfigStore to get the current version number
3. Pass both to RunLogEntry

```python
    # In cmd_run, after agent invocation and output extraction:
    # Save execution trace
    trace_path = ""
    try:
        trace_file = settings.traces_path / f"{traj.run_id}.json"
        if trace_file.exists():
            trace_path = str(trace_file)
    except Exception:
        pass

    # In the run log section, update RunLogEntry:
    from src.evolution.harness_config import HarnessConfigStore
    harness_store = HarnessConfigStore(settings.harness_config_path)
    harness_version = harness_store.get_latest_version()

    run_log.append(
        RunLogEntry(
            run_id=traj.run_id,
            task=task,
            output=output,
            classification=analysis["classification"],
            average_score=analysis["average_score"],
            grader_results=[asdict(g) for g in graders],
            prompt_version=current_version,
            harness_config_version=harness_version,
            trace_path=trace_path,
        )
    )
```

- [ ] **Step 5: Also score the harness config version in cmd_run**

After scoring the prompt version, add:

```python
    # Score the current harness config version
    if harness_version > 0:
        harness_store.update_score(harness_version, analysis["average_score"])
```

- [ ] **Step 6: Run all tests**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/evolution/run_log.py src/cli/commands.py
git commit -m "feat: capture trace paths and harness config version in run log"
```

---

### Task 4: Harness Optimizer — Evolve Middleware Parameters From Traces

**Files:**
- Create: `src/evolution/harness_optimizer.py`
- Test: `tests/unit/test_harness_optimizer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_harness_optimizer.py
"""Tests for harness parameter optimization."""

from unittest.mock import MagicMock, patch
from src.evolution.harness_config import HarnessConfig


class TestHarnessOptimizer:
    def test_build_harness_diagnosis_includes_trace_data(self):
        from src.evolution.harness_optimizer import build_harness_diagnosis
        from src.evolution.run_log import RunLogEntry

        entries = [
            RunLogEntry(
                run_id="r1",
                task="test quantum",
                output="report",
                classification="partial",
                average_score=0.72,
                grader_results=[
                    {"name": "task_completion", "score": 0.4, "passed": False,
                     "reasoning": "missed subtopics"},
                ],
                prompt_version=7,
                harness_config_version=1,
                trace_path="",
            ),
        ]
        diagnosis = build_harness_diagnosis(entries, HarnessConfig())
        assert "task_completion" in diagnosis
        assert "missed subtopics" in diagnosis
        assert "self_verification_max_retries: 2" in diagnosis

    def test_propose_harness_changes_returns_valid_config(self):
        from src.evolution.harness_optimizer import propose_harness_changes

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"self_verification_max_retries": 3, "time_budget_seconds": 240, '
                    '"reasoning": "More retries and tighter time budget"}'
        )
        current = HarnessConfig()
        new_cfg, reasoning = propose_harness_changes(mock_llm, "diagnosis text", current)
        assert new_cfg.self_verification_max_retries == 3
        assert new_cfg.time_budget_seconds == 240
        # Unchanged params stay at defaults
        assert new_cfg.loop_detection_max_similar == 3

    def test_propose_harness_changes_clamps_extreme_values(self):
        from src.evolution.harness_optimizer import propose_harness_changes

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"self_verification_max_retries": 100, "time_budget_seconds": 5}'
        )
        current = HarnessConfig()
        new_cfg, _ = propose_harness_changes(mock_llm, "diagnosis", current)
        assert new_cfg.self_verification_max_retries <= 10
        assert new_cfg.time_budget_seconds >= 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_harness_optimizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.evolution.harness_optimizer'`

- [ ] **Step 3: Implement harness_optimizer.py**

```python
# src/evolution/harness_optimizer.py
"""Harness parameter optimizer.

Reads execution traces and grading results to propose changes to
middleware parameters (retry counts, thresholds, time budgets, etc.).
Uses raw trace data for counterfactual diagnosis rather than summaries.
"""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evoagent.core.parsing import parse_llm_json
from src.evolution.harness_config import HarnessConfig
from src.evolution.run_log import RunLogEntry

logger = logging.getLogger(__name__)

# Bounds for parameters to prevent the optimizer from going wild
_PARAM_BOUNDS: dict[str, tuple[int | float, int | float]] = {
    "self_verification_max_retries": (0, 10),
    "self_verification_min_length": (100, 5000),
    "loop_detection_max_similar": (1, 20),
    "loop_detection_max_total": (3, 50),
    "loop_detection_max_file_edits": (1, 20),
    "loop_detection_max_repeated_tools": (1, 15),
    "time_budget_seconds": (60, 600),
    "reasoning_planning_calls": (1, 5),
}

_VALID_EFFORTS = {"low", "medium", "high"}

HARNESS_OPTIMIZER_PROMPT = (
    "You are optimizing the middleware harness parameters for a research agent.\n\n"
    "## Current Harness Configuration\n{current_config}\n\n"
    "## Diagnosis From Recent Runs\n{diagnosis}\n\n"
    "## Raw Execution Traces\n{trace_data}\n\n"
    "Based on the diagnosis and traces, propose parameter changes that would "
    "improve the agent's performance. Focus on the specific failure modes observed.\n\n"
    "Only change parameters where the traces show a clear signal. Leave others unchanged.\n\n"
    "Respond as JSON with the parameter names as keys and new values as values. "
    "Include a 'reasoning' key explaining your changes. Example:\n"
    '{"self_verification_max_retries": 3, "time_budget_seconds": 240, '
    '"reasoning": "Traces show verification catches issues but needs more retries"}\n\n'
    "Available parameters and their current values:\n{param_list}\n\n"
    "Respond with ONLY the JSON object, nothing else."
)


def _load_trace(trace_path: str) -> str:
    """Load a trace file and return a compact summary."""
    if not trace_path:
        return ""
    path = Path(trace_path)
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text())
        parts = [f"Duration: {data.get('duration_seconds', '?')}s"]
        parts.append(f"Steps: {data.get('step_count', '?')}")
        for step in data.get("steps", [])[:20]:
            step_type = step.get("type", "unknown")
            if step_type == "tool_call":
                parts.append(f"  TOOL: {step.get('name', '?')} args={step.get('args_preview', '')[:100]}")
                output = step.get("output_preview", "")
                if any(kw in output.lower() for kw in ["error", "fail", "timeout", "quota"]):
                    parts.append(f"    ERROR: {output[:200]}")
            elif step_type == "model_call":
                tool_calls = step.get("tool_calls", [])
                if tool_calls:
                    parts.append(f"  MODEL: called {[tc.get('name') for tc in tool_calls]}")
            elif step_type == "truncated":
                parts.append(f"  ... {step.get('removed_steps', 0)} steps truncated ...")
        return "\n".join(parts)
    except Exception:
        return ""


def build_harness_diagnosis(
    entries: list[RunLogEntry],
    current_config: HarnessConfig,
) -> str:
    """Build a diagnostic summary from recent run entries.

    Includes per-dimension failure patterns and the current config so the
    optimizer can see what's tunable.
    """
    lines: list[str] = []
    lines.append(f"Runs analyzed: {len(entries)}")

    # Per-dimension failure rates
    from collections import defaultdict
    dim_fails: dict[str, int] = defaultdict(int)
    dim_reasons: dict[str, list[str]] = defaultdict(list)

    for entry in entries:
        for g in entry.grader_results:
            if not g.get("passed", True):
                dim_fails[g["name"]] += 1
                reason = g.get("reasoning", "")[:100]
                if reason and len(dim_reasons[g["name"]]) < 5:
                    dim_reasons[g["name"]].append(reason)

    if dim_fails:
        lines.append("\nFailing dimensions:")
        for name, count in sorted(dim_fails.items(), key=lambda x: -x[1]):
            rate = count / len(entries)
            lines.append(f"  {name}: {count}/{len(entries)} ({rate:.0%} fail rate)")
            for reason in dim_reasons[name][:3]:
                lines.append(f"    - {reason}")

    # Current config for reference
    lines.append("\nCurrent harness parameters:")
    for key, value in current_config.to_dict().items():
        lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def propose_harness_changes(
    llm: BaseChatModel,
    diagnosis: str,
    current_config: HarnessConfig,
    trace_data: str = "",
) -> tuple[HarnessConfig, str]:
    """Use LLM to propose harness parameter changes based on diagnosis.

    Returns (new_config, reasoning).
    """
    config_dict = current_config.to_dict()
    param_list = "\n".join(f"  {k}: {v}" for k, v in config_dict.items())

    prompt = HARNESS_OPTIMIZER_PROMPT.format(
        current_config=json.dumps(config_dict, indent=2),
        diagnosis=diagnosis,
        trace_data=trace_data or "No trace data available.",
        param_list=param_list,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = parse_llm_json(response.content)
    if not parsed:
        return current_config, "LLM returned unparseable response"

    reasoning = parsed.pop("reasoning", "")

    # Apply changes to a copy of the current config
    new_dict = config_dict.copy()
    for key, value in parsed.items():
        if key not in config_dict:
            continue
        # Clamp numeric values to bounds
        if key in _PARAM_BOUNDS:
            lo, hi = _PARAM_BOUNDS[key]
            value = max(lo, min(hi, value))
        # Validate effort strings
        if "effort" in key and isinstance(value, str):
            if value not in _VALID_EFFORTS:
                continue
        new_dict[key] = value

    return HarnessConfig.from_dict(new_dict), reasoning


def optimize_harness(
    llm: BaseChatModel,
    entries: list[RunLogEntry],
    config_store: "HarnessConfigStore",
) -> int:
    """Run the full harness optimization pipeline.

    Returns the new config version number.
    """
    from src.evolution.harness_config import HarnessConfigStore

    current = config_store.load_best()

    # Build diagnosis
    diagnosis = build_harness_diagnosis(entries, current)

    # Collect trace data from entries (up to 5 traces, ~2K chars each)
    trace_parts: list[str] = []
    for entry in entries[:5]:
        if entry.trace_path:
            trace = _load_trace(entry.trace_path)
            if trace:
                trace_parts.append(f"### Run: {entry.task[:60]}\n{trace}")

    trace_data = "\n\n".join(trace_parts) if trace_parts else ""

    # Propose changes
    new_config, reasoning = propose_harness_changes(
        llm, diagnosis, current, trace_data
    )

    # Only save if something actually changed
    if new_config.to_dict() == current.to_dict():
        logger.info("Harness optimizer: no changes proposed")
        return config_store.get_latest_version()

    logger.info("Harness optimizer: %s", reasoning[:200])
    version = config_store.save(new_config, score=None)
    return version
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_harness_optimizer.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/evolution/harness_optimizer.py tests/unit/test_harness_optimizer.py
git commit -m "feat: add harness optimizer that evolves middleware parameters from traces"
```

---

### Task 5: Extend Daemon to Optimize Harness + Use Raw Traces

**Files:**
- Modify: `src/evolution/daemon.py:66-150` (add harness optimization step + raw trace loading)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_daemon_harness.py
"""Tests for daemon harness optimization integration."""

from unittest.mock import MagicMock, patch
from pathlib import Path

from src.evolution.run_log import RunLogEntry


class TestDaemonHarnessOptimization:
    def test_evolution_cycle_includes_harness_optimization(self):
        from src.evolution.daemon import _run_evolution_cycle
        from src.config.settings import Settings

        entry = RunLogEntry(
            run_id="r1",
            task="test",
            output="report",
            classification="partial",
            average_score=0.72,
            grader_results=[
                {"name": "task_completion", "score": 0.4, "passed": False,
                 "reasoning": "missed"},
            ],
            prompt_version=7,
            harness_config_version=1,
        )

        with patch("src.evolution.daemon.create_llm") as mock_llm_factory, \
             patch("src.evolution.daemon.optimize_prompt") as mock_opt, \
             patch("src.evolution.daemon.extract_skills_from_batch") as mock_skills, \
             patch("src.evolution.daemon.deduplicate_semantic") as mock_dedup, \
             patch("src.evolution.daemon.optimize_harness") as mock_harness:
            mock_llm_factory.return_value = MagicMock()
            mock_opt.return_value = 8
            mock_skills.return_value = []
            mock_dedup.return_value = 0
            mock_harness.return_value = 2

            settings = MagicMock(spec=Settings)
            settings.skills_path = Path("/tmp/skills")
            settings.prompts_path = Path("/tmp/prompts")
            settings.memory_path = Path("/tmp/memory")
            settings.harness_config_path = Path("/tmp/harness")
            settings.compression_similarity_threshold = 0.7

            result = _run_evolution_cycle(settings, [entry])
            mock_harness.assert_called_once()
            assert "harness_changed" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_daemon_harness.py -v`
Expected: FAIL — `optimize_harness` not yet imported in daemon

- [ ] **Step 3: Update daemon.py to include harness optimization**

In `src/evolution/daemon.py`, add the import and harness optimization step:

Add to imports:
```python
from src.evolution.harness_config import HarnessConfigStore
from src.evolution.harness_optimizer import optimize_harness
```

In `_run_evolution_cycle`, add after the memory compression step (step 4) and before the return:

```python
    # 5. Harness optimization
    try:
        harness_store = HarnessConfigStore(settings.harness_config_path)
        old_harness = harness_store.get_latest_version()
        new_harness = optimize_harness(llm, entries, harness_store)
        if new_harness != old_harness:
            logger.info("Harness config upgraded: v%d -> v%d", old_harness, new_harness)
            result["harness_changed"] = True
        else:
            result["harness_changed"] = False
    except Exception as exc:
        logger.error("Harness optimization failed: %s", exc)
        result["harness_changed"] = False
```

Also add `"harness_changed": False` to the initial `result` dict.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_daemon_harness.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/evolution/daemon.py tests/unit/test_daemon_harness.py
git commit -m "feat: daemon now optimizes harness parameters alongside prompts"
```

---

### Task 6: Enable Environment Bootstrapping and Time Budget by Default

**Files:**
- Modify: `src/agent/deep_agent.py:158-164` (enable detect_env and time budget)
- Modify: `evoagent/harness/middleware.py:221-277` (ContextAssemblyMiddleware — ensure detect_env works)

This is a configuration-only change. The middleware already supports `detect_env=True` and `TimeBudgetMiddleware`, but they were not wired in by default.

- [ ] **Step 1: Verify detect_env and time budget work in middleware**

Run a quick smoke test:

```bash
PYTHONPATH=. .venv/bin/python -c "
from evoagent.harness.middleware import ContextAssemblyMiddleware, detect_environment
env = detect_environment()
print('Working dir:', env.get('working_directory'))
print('Tools found:', len(env.get('available_tools', [])))
print('Dir entries:', len(env.get('directory_listing', [])))
"
```

Expected: Prints working directory, available tools, and directory listing without errors.

- [ ] **Step 2: Update HarnessConfig defaults to enable env detection**

The `HarnessConfig` already has `context_assembly_detect_env: bool = True` from Task 1. Verify by running:

```bash
PYTHONPATH=. .venv/bin/python -c "
from src.evolution.harness_config import HarnessConfig
cfg = HarnessConfig()
print('detect_env:', cfg.context_assembly_detect_env)
print('time_budget:', cfg.time_budget_seconds)
"
```

Expected: `detect_env: True`, `time_budget: 300`

- [ ] **Step 3: Run full test suite to confirm no regressions**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/evolution/harness_config.py
git commit -m "feat: enable environment detection and time budget by default via HarnessConfig"
```

---

### Task 7: Remove Explicit `evolve` Dependency — Daemon Is the Primary Loop

**Files:**
- Modify: `src/cli/commands.py` (update help text, make daemon the recommended path)
- Modify: `src/evolution/daemon.py` (add auto-start capability when first run completes)

- [ ] **Step 1: Add auto-daemon hint to cmd_run output**

In `src/cli/commands.py`, at the end of `cmd_run`, after the reflection output, add:

```python
    # Hint about daemon if not running
    run_log = RunLog(settings.evolution_state_path / "run_log.jsonl")
    pending = run_log.count_unprocessed()
    if pending >= 3:
        print(f"\n  {pending} unprocessed runs. Start the daemon to auto-evolve:")
        print("    python -m src evolve-daemon")
```

- [ ] **Step 2: Update daemon logging to show what changed**

In `src/evolution/daemon.py`, update the cycle completion log in `run_daemon` to show the daemon's effect more clearly:

```python
                logger.info(
                    "Daemon Cycle %d complete: prompt_changed=%s, "
                    "harness_changed=%s, skills=%d, failure_skills=%d",
                    cycle_count,
                    result["prompt_changed"],
                    result.get("harness_changed", False),
                    result["skills_extracted"],
                    result["failure_skills_created"],
                )
```

- [ ] **Step 3: Run full test suite**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/cli/commands.py src/evolution/daemon.py
git commit -m "feat: daemon auto-evolve hints and harness change logging"
```

---

### Task 8: Integration Test — Full Single-Run → Daemon → Improved Next Run

**Files:**
- Create: `tests/unit/test_continuous_evolution.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/unit/test_continuous_evolution.py
"""Integration test for the continuous evolution pipeline.

Verifies that single run data flows through the daemon and produces
improved harness config and prompt versions.
"""

import json
from pathlib import Path
from dataclasses import asdict

from src.evolution.harness_config import HarnessConfig, HarnessConfigStore
from src.evolution.run_log import RunLog, RunLogEntry
from src.agent.prompt_store import PromptStore


class TestContinuousEvolutionPipeline:
    def test_run_log_entry_roundtrip(self):
        """RunLogEntry serializes and deserializes with all new fields."""
        entry = RunLogEntry(
            run_id="test-001",
            task="test quantum",
            output="report...",
            classification="partial",
            average_score=0.78,
            grader_results=[
                {"name": "task_completion", "score": 0.4, "passed": False, "reasoning": "missed"},
                {"name": "efficiency", "score": 0.88, "passed": True, "reasoning": "ok"},
            ],
            prompt_version=7,
            harness_config_version=2,
            trace_path="/tmp/trace.json",
        )
        d = entry.to_dict()
        restored = RunLogEntry(**d)
        assert restored.harness_config_version == 2
        assert restored.trace_path == "/tmp/trace.json"

    def test_harness_config_store_lifecycle(self, tmp_path):
        """Full lifecycle: save, score, load best."""
        store = HarnessConfigStore(tmp_path / "harness")

        # Save initial config
        cfg1 = HarnessConfig()
        store.save(cfg1, score=0.70)

        # Save evolved config
        cfg2 = HarnessConfig(self_verification_max_retries=3, time_budget_seconds=240)
        store.save(cfg2, score=0.82)

        # Best should be cfg2
        best = store.load_best()
        assert best.self_verification_max_retries == 3
        assert best.time_budget_seconds == 240

        # Incremental scoring
        store.update_score(2, 0.60)
        # Now score = (0.82 + 0.60) / 2 = 0.71
        data = json.loads((tmp_path / "harness" / "v0002.json").read_text())
        assert abs(data["score"] - 0.71) < 0.01

        # cfg1 is now best again (0.70 > 0.71 is false, but 0.70 < 0.71)
        # Actually 0.71 > 0.70, so cfg2 is still best
        best = store.load_best()
        assert best.time_budget_seconds == 240

    def test_prompt_and_harness_both_scored_from_runs(self, tmp_path):
        """Both prompt and harness config accumulate scores from run log entries."""
        prompt_store = PromptStore(tmp_path / "prompts")
        prompt_store.add_version("test prompt", score=None)

        harness_store = HarnessConfigStore(tmp_path / "harness")
        harness_store.save(HarnessConfig(), score=None)

        # Simulate 3 run scores
        for score in [0.75, 0.80, 0.70]:
            prompt_store.update_score(1, score)
            harness_store.update_score(1, score)

        # Both should have incremental average = 0.75
        p_data = json.loads((tmp_path / "prompts" / "v0001.json").read_text())
        h_data = json.loads((tmp_path / "harness" / "v0001.json").read_text())
        assert abs(p_data["score"] - 0.75) < 0.01
        assert abs(h_data["score"] - 0.75) < 0.01

    def test_run_log_with_harness_fields(self, tmp_path):
        """Run log persists and reads harness-related fields."""
        log = RunLog(tmp_path / "run_log.jsonl")

        log.append(RunLogEntry(
            run_id="r1", task="test", output="out",
            classification="partial", average_score=0.72,
            grader_results=[], prompt_version=7,
            harness_config_version=2, trace_path="/tmp/trace.json",
        ))

        entries = log.read_all()
        assert len(entries) == 1
        assert entries[0].harness_config_version == 2
        assert entries[0].trace_path == "/tmp/trace.json"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/unit/test_continuous_evolution.py -v`
Expected: All 4 tests PASS (these test the already-implemented components working together)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_continuous_evolution.py
git commit -m "test: add integration tests for continuous harness + prompt evolution pipeline"
```

---

### Task 9: Final — Run Full Test Suite and Push

- [ ] **Step 1: Run the complete test suite**

Run: `PYTHONPATH=. .venv/bin/python -m pytest tests/ -v`
Expected: All tests PASS (259 existing + ~15 new)

- [ ] **Step 2: Verify single run works end-to-end**

Run: `python -m src run "What is CRISPR?" 2>&1 | tail -20`
Expected: Output includes GRADING section + "Reflection stored" + harness config scoring

- [ ] **Step 3: Verify daemon starts cleanly**

Run: `timeout 5 python -m src evolve-daemon --interval 2 --min-runs 100 2>&1 || true`
Expected: Shows "Evolution daemon started" and exits cleanly after timeout

- [ ] **Step 4: Commit all remaining changes**

```bash
git add -A
git commit -m "feat: continuous harness and context evolution from user interactions

Complete pipeline: every run grades, scores prompt + harness config,
reflects into memory, and appends to run log. Background daemon watches
run log and evolves harness parameters, prompts, and skills.

No explicit 'evolve' call needed — the agent self-improves from usage."
```

- [ ] **Step 5: Push to remote**

```bash
git push
```
