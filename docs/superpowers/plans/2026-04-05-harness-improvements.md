# Harness Layer Improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close 7 gaps identified in comparison with Meta-Harness, LangChain Harness Engineering, and Karpathy's autoresearch.

**Architecture:** All improvements are additive to `evoagent/harness/middleware.py` (4 existing + 2 new middleware classes), `evoagent/evolution/error_analyzer.py` (new), `evoagent/evolution/sleep_review.py` (enhanced), and `evoagent/evolution/prompt_optimizer.py` (enhanced). Each improvement is independently testable.

**Tech Stack:** Python 3.12+, langchain AgentMiddleware, ThreadPoolExecutor

**Spec:** `docs/superpowers/specs/2026-04-05-harness-improvements-design.md`

---

## File Structure

### Modified Files

- `evoagent/harness/middleware.py` — Tasks 1, 2, 3, 5, 6 (env bootstrapping, time budget, file-edit loop, reasoning sandwich, task verification)
- `evoagent/harness/builder.py` — Task 7 (update default stack with new middleware + params)
- `evoagent/harness/__init__.py` — Task 7 (export new classes)
- `evoagent/evolution/prompt_optimizer.py` — Task 4 (use deep error analysis)
- `evoagent/evolution/sleep_review.py` — Task 8 (contradiction detection)

### New Files

- `evoagent/evolution/error_analyzer.py` — Task 4 (parallel deep failure analysis)
- `tests/evoagent/integration/test_env_bootstrapping.py` — Task 1
- `tests/evoagent/integration/test_time_budget.py` — Task 2
- `tests/evoagent/integration/test_file_edit_loop.py` — Task 3
- `tests/evoagent/integration/test_error_analyzer.py` — Task 4
- `tests/evoagent/integration/test_reasoning_sandwich.py` — Task 5
- `tests/evoagent/integration/test_task_verification.py` — Task 6
- `tests/evoagent/integration/test_builder_updated.py` — Task 7
- `tests/evoagent/integration/test_contradiction_detection.py` — Task 8

---

## Task 1: Environment Bootstrapping in ContextAssemblyMiddleware

**Files:**
- Modify: `evoagent/harness/middleware.py`
- Create: `tests/evoagent/integration/test_env_bootstrapping.py`

- [ ] **Step 1: Write failing tests**

Create `tests/evoagent/integration/test_env_bootstrapping.py`:

```python
"""Tests for environment bootstrapping in ContextAssemblyMiddleware."""

from evoagent.harness.middleware import detect_environment


def test_detect_environment_returns_dict(tmp_path):
    result = detect_environment(working_dir=tmp_path)
    assert isinstance(result, dict)
    assert "working_directory" in result
    assert "directory_listing" in result
    assert "available_tools" in result


def test_detect_environment_lists_files(tmp_path):
    (tmp_path / "README.md").write_text("hello")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "input.csv").write_text("a,b")
    result = detect_environment(working_dir=tmp_path)
    assert "README.md" in result["directory_listing"]
    assert "data/" in result["directory_listing"]


def test_detect_environment_caps_listing(tmp_path):
    for i in range(100):
        (tmp_path / f"file_{i}.txt").write_text("x")
    result = detect_environment(working_dir=tmp_path, max_entries=50)
    lines = result["directory_listing"].strip().split("\n")
    assert len(lines) <= 51  # 50 entries + possible "... and N more"


def test_detect_environment_finds_tools(tmp_path):
    result = detect_environment(working_dir=tmp_path)
    # python3 should be available in test environment
    assert any("python" in t for t in result["available_tools"])


def test_format_environment_context():
    from evoagent.harness.middleware import format_environment_context
    env = {
        "working_directory": "/tmp/test",
        "directory_listing": "README.md\ndata/",
        "available_tools": ["python3", "git"],
    }
    text = format_environment_context(env)
    assert "Working directory:" in text
    assert "README.md" in text
    assert "python3" in text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
PYTHONPATH=. python -m pytest tests/evoagent/integration/test_env_bootstrapping.py -v
```

- [ ] **Step 3: Implement `detect_environment` and `format_environment_context`**

Add to `evoagent/harness/middleware.py` before the `ContextAssemblyMiddleware` class:

```python
# --- Environment Detection ---

_DEFAULT_TOOL_NAMES = ["python3", "python", "node", "npm", "curl", "git", "make", "gcc", "java", "go"]


def detect_environment(
    working_dir: Path | None = None,
    max_entries: int = 50,
    max_depth: int = 2,
) -> dict[str, Any]:
    """Detect system environment: directory structure, available tools."""
    import shutil
    import subprocess

    cwd = Path(working_dir) if working_dir else Path.cwd()
    result: dict[str, Any] = {"working_directory": str(cwd)}

    # Directory listing (capped)
    entries: list[str] = []
    try:
        for item in sorted(cwd.iterdir()):
            if item.name.startswith("."):
                continue
            if item.is_dir():
                entries.append(f"{item.name}/")
                if max_depth >= 2:
                    try:
                        for child in sorted(item.iterdir())[:10]:
                            if not child.name.startswith("."):
                                suffix = "/" if child.is_dir() else ""
                                entries.append(f"  {child.name}{suffix}")
                    except PermissionError:
                        pass
            else:
                entries.append(item.name)
            if len(entries) >= max_entries:
                remaining = sum(1 for _ in cwd.iterdir()) - max_entries
                if remaining > 0:
                    entries.append(f"... and {remaining} more")
                break
    except PermissionError:
        entries.append("(permission denied)")
    result["directory_listing"] = "\n".join(entries)

    # Available tools
    tools: list[str] = []
    for tool_name in _DEFAULT_TOOL_NAMES:
        path = shutil.which(tool_name)
        if path:
            # Get version for key tools
            version = ""
            try:
                out = subprocess.run(
                    [tool_name, "--version"], capture_output=True, text=True, timeout=3
                )
                first_line = (out.stdout or out.stderr).strip().split("\n")[0]
                version = f" ({first_line[:60]})" if first_line else ""
            except Exception:
                pass
            tools.append(f"{tool_name}{version}")
    result["available_tools"] = tools

    return result


def format_environment_context(env: dict[str, Any]) -> str:
    """Format environment detection results as markdown context."""
    parts = [
        f"## System Environment",
        f"- Working directory: {env.get('working_directory', '?')}",
        f"- Directory structure:\n```\n{env.get('directory_listing', '(empty)')}\n```",
    ]
    tools = env.get("available_tools", [])
    if tools:
        parts.append(f"- Available tools: {', '.join(tools)}")
    return "\n".join(parts)
```

- [ ] **Step 4: Update `ContextAssemblyMiddleware.__init__` and `before_model`**

```python
class ContextAssemblyMiddleware(AgentMiddleware):
    """Enriches first model call with environment context."""

    tools: tuple[BaseTool, ...] = ()

    def __init__(
        self,
        skills_dir: Path | None = None,
        detect_env: bool = False,
        working_dir: Path | None = None,
    ) -> None:
        self._skills_dir = skills_dir
        self._detect_env = detect_env
        self._working_dir = working_dir
        self._first_call = True

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if not self._first_call:
            return None
        self._first_call = False

        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        if not messages:
            return None

        context_parts: list[str] = []

        # Environment bootstrapping (Meta-Harness pattern)
        if self._detect_env:
            try:
                env = detect_environment(working_dir=self._working_dir)
                context_parts.append(format_environment_context(env))
            except Exception as exc:
                logger.debug("Environment detection failed: %s", exc)

        # Skill discovery
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
```

- [ ] **Step 5: Run tests**

```bash
PYTHONPATH=. python -m pytest tests/evoagent/integration/test_env_bootstrapping.py -v
```

- [ ] **Step 6: Commit**

```bash
git add evoagent/harness/middleware.py tests/evoagent/integration/test_env_bootstrapping.py
git commit -m "feat(evoagent): add environment bootstrapping to ContextAssemblyMiddleware

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: TimeBudgetMiddleware

**Files:**
- Modify: `evoagent/harness/middleware.py`
- Create: `tests/evoagent/integration/test_time_budget.py`

- [ ] **Step 1: Write failing tests**

Create `tests/evoagent/integration/test_time_budget.py`:

```python
"""Tests for TimeBudgetMiddleware."""

import time

from evoagent.harness.middleware import TimeBudgetMiddleware


def test_no_warning_before_threshold():
    mw = TimeBudgetMiddleware(budget_seconds=100, warn_at=[0.6, 0.85])
    mw.start()
    # Just started, no warning
    warning = mw.get_warning()
    assert warning is None


def test_warning_at_60_percent():
    mw = TimeBudgetMiddleware(budget_seconds=10, warn_at=[0.6, 0.85])
    mw._start_time = time.monotonic() - 7  # 70% elapsed
    warning = mw.get_warning()
    assert warning is not None
    assert "synthesiz" in warning.lower()


def test_warning_at_85_percent():
    mw = TimeBudgetMiddleware(budget_seconds=10, warn_at=[0.6, 0.85])
    mw._start_time = time.monotonic() - 9  # 90% elapsed
    mw._warnings_fired.add(0.6)  # 60% already fired
    warning = mw.get_warning()
    assert warning is not None
    assert "stop" in warning.lower() or "urgent" in warning.lower()


def test_warning_fires_only_once():
    mw = TimeBudgetMiddleware(budget_seconds=10, warn_at=[0.6, 0.85])
    mw._start_time = time.monotonic() - 7  # 70% elapsed
    w1 = mw.get_warning()
    assert w1 is not None
    w2 = mw.get_warning()
    assert w2 is None  # already fired


def test_elapsed_fraction():
    mw = TimeBudgetMiddleware(budget_seconds=100)
    mw._start_time = time.monotonic() - 50
    frac = mw.elapsed_fraction()
    assert 0.45 <= frac <= 0.55
```

- [ ] **Step 2: Implement `TimeBudgetMiddleware`**

Add to `evoagent/harness/middleware.py`:

```python
# --- Time Budget ---

class TimeBudgetMiddleware(AgentMiddleware):
    """Injects time warnings to push the agent toward completion."""

    tools: tuple[BaseTool, ...] = ()

    def __init__(
        self,
        budget_seconds: int = 300,
        warn_at: list[float] | None = None,
    ) -> None:
        self._budget = budget_seconds
        self._warn_at = sorted(warn_at or [0.6, 0.85])
        self._start_time: float = 0.0
        self._warnings_fired: set[float] = set()

    def start(self) -> None:
        self._start_time = time.monotonic()

    def before_agent(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        if not self._start_time:
            self._start_time = time.monotonic()
        return None

    def elapsed_fraction(self) -> float:
        if not self._start_time:
            return 0.0
        return (time.monotonic() - self._start_time) / self._budget

    def get_warning(self) -> str | None:
        """Check if a warning should fire. Returns warning text or None."""
        frac = self.elapsed_fraction()
        for threshold in self._warn_at:
            if frac >= threshold and threshold not in self._warnings_fired:
                self._warnings_fired.add(threshold)
                elapsed = int(time.monotonic() - self._start_time)
                if threshold >= 0.85:
                    return (
                        f"\u23f1 URGENT: {elapsed}s of {self._budget}s used ({frac:.0%}). "
                        "Stop all searches. Write your final report NOW using what you have."
                    )
                return (
                    f"\u23f1 Time check: {elapsed}s of {self._budget}s used ({frac:.0%}). "
                    "Start synthesizing your findings."
                )
        return None

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        result = handler(request)
        warning = self.get_warning()
        if warning and hasattr(result, "content") and isinstance(result.content, str):
            from langchain_core.messages import ToolMessage
            result = ToolMessage(
                content=result.content + f"\n\n{warning}",
                tool_call_id=result.tool_call_id,
                name=getattr(result, "name", ""),
            )
        return result
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=. python -m pytest tests/evoagent/integration/test_time_budget.py -v
```

- [ ] **Step 4: Commit**

```bash
git add evoagent/harness/middleware.py tests/evoagent/integration/test_time_budget.py
git commit -m "feat(evoagent): add TimeBudgetMiddleware for time pressure injection

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: File-Edit Loop Detection

**Files:**
- Modify: `evoagent/harness/middleware.py`
- Create: `tests/evoagent/integration/test_file_edit_loop.py`

- [ ] **Step 1: Write failing tests**

Create `tests/evoagent/integration/test_file_edit_loop.py`:

```python
"""Tests for file-edit loop detection."""

from evoagent.harness.middleware import LoopDetectionMiddleware


def test_file_edit_tracking():
    mw = LoopDetectionMiddleware(max_similar=3, max_total=12, max_file_edits=3)
    assert mw.should_warn_file_edit("main.py") is False
    assert mw.should_warn_file_edit("main.py") is False
    assert mw.should_warn_file_edit("main.py") is True  # 3rd edit


def test_different_files_no_warning():
    mw = LoopDetectionMiddleware(max_file_edits=3)
    assert mw.should_warn_file_edit("a.py") is False
    assert mw.should_warn_file_edit("b.py") is False
    assert mw.should_warn_file_edit("c.py") is False


def test_file_edit_warning_fires_once():
    mw = LoopDetectionMiddleware(max_file_edits=2)
    mw.should_warn_file_edit("main.py")  # 1st
    assert mw.should_warn_file_edit("main.py") is True   # 2nd -> warn
    assert mw.should_warn_file_edit("main.py") is False  # 3rd -> already warned


def test_repeated_tool_tracking():
    mw = LoopDetectionMiddleware(max_repeated_tools=3)
    assert mw.should_warn_repeated_tool("bash", "ls /tmp") is False
    assert mw.should_warn_repeated_tool("bash", "ls /tmp/foo") is False
    assert mw.should_warn_repeated_tool("bash", "ls /tmp/bar") is True  # 3rd similar
```

- [ ] **Step 2: Implement enhancements to `LoopDetectionMiddleware`**

Update the class in `evoagent/harness/middleware.py`:

```python
class LoopDetectionMiddleware(AgentMiddleware):
    """Detects repetitive search queries, file edits, and tool calls."""

    tools: tuple[BaseTool, ...] = ()

    def __init__(
        self,
        max_similar: int = 3,
        max_total: int = 12,
        max_file_edits: int = 5,
        max_repeated_tools: int = 4,
    ) -> None:
        self._max_similar = max_similar
        self._max_total = max_total
        self._max_file_edits = max_file_edits
        self._max_repeated_tools = max_repeated_tools
        self._queries: list[str] = []
        self._warned = False
        self._file_edit_counts: dict[str, int] = {}
        self._file_edit_warned: set[str] = set()
        self._tool_calls: list[tuple[str, str]] = []  # (tool_name, args_str)
        self._tool_warned = False

    def should_warn(self, query: str) -> bool:
        """Track a search query and return True if warning should fire."""
        similar_count = sum(1 for q in self._queries if is_similar_query(q, query))
        self._queries.append(query)
        return (similar_count >= self._max_similar or len(self._queries) >= self._max_total) and not self._warned

    def should_warn_file_edit(self, file_path: str) -> bool:
        """Track a file edit and return True if warning should fire."""
        self._file_edit_counts[file_path] = self._file_edit_counts.get(file_path, 0) + 1
        count = self._file_edit_counts[file_path]
        if count >= self._max_file_edits and file_path not in self._file_edit_warned:
            self._file_edit_warned.add(file_path)
            return True
        return False

    def should_warn_repeated_tool(self, tool_name: str, args_str: str) -> bool:
        """Track a tool call and return True if too many similar calls."""
        similar = sum(
            1 for tn, a in self._tool_calls
            if tn == tool_name and is_similar_query(a, args_str)
        )
        self._tool_calls.append((tool_name, args_str))
        if similar >= self._max_repeated_tools and not self._tool_warned:
            self._tool_warned = True
            return True
        return False

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        args_str = str(args)

        result = handler(request)

        # Check file edits
        warning = None
        file_path = args.get("file_path", args.get("path", args.get("filename", "")))
        if file_path and tool_name in ("write_file", "edit_file", "create_file", "patch"):
            if self.should_warn_file_edit(str(file_path)):
                count = self._file_edit_counts.get(str(file_path), 0)
                warning = (
                    f"\n\n\u26a0 You've edited {file_path} {count} times. "
                    "Consider reconsidering your approach or moving on."
                )

        # Check repeated tools
        if not warning and self.should_warn_repeated_tool(tool_name, args_str):
            warning = (
                f"\n\n\u26a0 LOOP DETECTED: Repeated similar calls to {tool_name}. "
                "Consider a different approach."
            )

        if warning and hasattr(result, "content") and isinstance(result.content, str):
            from langchain_core.messages import ToolMessage
            result = ToolMessage(
                content=result.content + warning,
                tool_call_id=result.tool_call_id,
                name=getattr(result, "name", tool_name),
            )

        return result
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=. python -m pytest tests/evoagent/integration/test_file_edit_loop.py -v
```

- [ ] **Step 4: Commit**

```bash
git add evoagent/harness/middleware.py tests/evoagent/integration/test_file_edit_loop.py
git commit -m "feat(evoagent): extend LoopDetection with file-edit and repeated-tool tracking

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Parallel Error Analysis

**Files:**
- Create: `evoagent/evolution/error_analyzer.py`
- Modify: `evoagent/evolution/prompt_optimizer.py`
- Create: `tests/evoagent/integration/test_error_analyzer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/evoagent/integration/test_error_analyzer.py`:

```python
"""Tests for parallel deep error analysis."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from evoagent.evolution.error_analyzer import analyze_failures_deep


def test_returns_analysis_per_failure():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content='{"root_cause": "agent searched wrong terms", '
        '"counterfactual": "should have searched for X", '
        '"suggested_guidance": "Always search for specific terms"}'
    )
    failures = [
        {"task": "test task", "output": "bad output", "average_score": 0.3,
         "grader_results": [{"name": "quality", "score": 0.2, "passed": False, "reasoning": "poor"}]},
    ]
    results = analyze_failures_deep(llm, failures)
    assert len(results) == 1
    assert "root_cause" in results[0]
    assert "counterfactual" in results[0]
    assert "suggested_guidance" in results[0]


def test_empty_failures_returns_empty():
    llm = MagicMock()
    results = analyze_failures_deep(llm, [])
    assert results == []


def test_handles_llm_error_gracefully():
    llm = MagicMock()
    llm.invoke.side_effect = Exception("LLM broke")
    failures = [
        {"task": "test", "output": "bad", "average_score": 0.2, "grader_results": []},
    ]
    results = analyze_failures_deep(llm, failures)
    assert len(results) == 1
    assert "error" in results[0].get("root_cause", "").lower() or results[0].get("root_cause") == "analysis_failed"


def test_caps_at_max_parallel():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content='{"root_cause": "x", "counterfactual": "y", "suggested_guidance": "z"}'
    )
    failures = [{"task": f"task {i}", "output": "x", "average_score": 0.2, "grader_results": []} for i in range(10)]
    results = analyze_failures_deep(llm, failures, max_parallel=3)
    # Should analyze at most 5 (default cap), not all 10
    assert len(results) <= 5
```

- [ ] **Step 2: Implement `evoagent/evolution/error_analyzer.py`**

```python
"""Parallel deep error analysis for failed trajectories.

Spawns parallel LLM calls to identify root causes and counterfactuals
for each failure, providing richer context to the prompt optimizer.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evoagent.core.parsing import parse_llm_json

logger = logging.getLogger(__name__)

ERROR_ANALYSIS_PROMPT = (
    "Analyze this failed agent trajectory and identify what went wrong.\n\n"
    "## Task\n{task}\n\n"
    "## Agent Output (preview)\n{output}\n\n"
    "## Grader Results\n{grader_results}\n\n"
    "## Execution Trace\n{trace}\n\n"
    "Identify:\n"
    "1. root_cause: The specific decision point where the agent went wrong\n"
    "2. counterfactual: What the agent should have done instead\n"
    "3. suggested_guidance: A concrete instruction for the system prompt to prevent this\n\n"
    'Respond as JSON: {{"root_cause": "...", "counterfactual": "...", "suggested_guidance": "..."}}'
)

MAX_FAILURES_TO_ANALYZE = 5


def _analyze_single_failure(
    llm: BaseChatModel,
    failure: dict[str, Any],
    trace: str = "",
) -> dict[str, Any]:
    """Analyze a single failure via LLM."""
    try:
        grader_text = "\n".join(
            f"- [{g.get('name', '?')}] score={g.get('score', '?')}: {g.get('reasoning', '')[:150]}"
            for g in failure.get("grader_results", [])
            if isinstance(g, dict)
        )
        prompt = ERROR_ANALYSIS_PROMPT.format(
            task=failure.get("task", "")[:500],
            output=failure.get("output", "")[:2000],
            grader_results=grader_text or "No grader details available",
            trace=trace[:1000] or "No trace available",
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = parse_llm_json(response.content)
        return {
            "task": failure.get("task", "")[:200],
            "root_cause": parsed.get("root_cause", "unknown"),
            "counterfactual": parsed.get("counterfactual", "unknown"),
            "suggested_guidance": parsed.get("suggested_guidance", ""),
        }
    except Exception as exc:
        logger.error("Error analysis failed for task '%s': %s", failure.get("task", "?")[:50], exc)
        return {
            "task": failure.get("task", "")[:200],
            "root_cause": "analysis_failed",
            "counterfactual": str(exc),
            "suggested_guidance": "",
        }


def _load_trace_for_task(task: str, traces_dir: Path | None) -> str:
    """Load the most recent trace matching a task."""
    if not traces_dir or not traces_dir.exists():
        return ""
    for trace_file in sorted(traces_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(trace_file.read_text())
            if data.get("task", "")[:100] == task[:100]:
                steps = data.get("steps", [])
                return json.dumps(steps[:10], indent=2, default=str)[:1000]
        except Exception:
            continue
    return ""


def analyze_failures_deep(
    llm: BaseChatModel,
    failed_analyses: list[dict[str, Any]],
    traces_dir: Path | None = None,
    max_parallel: int = 3,
    max_failures: int = MAX_FAILURES_TO_ANALYZE,
) -> list[dict[str, Any]]:
    """Run parallel deep error analysis on failed trajectories.

    Returns list of {task, root_cause, counterfactual, suggested_guidance}.
    """
    if not failed_analyses:
        return []

    to_analyze = failed_analyses[:max_failures]
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for failure in to_analyze:
            trace = _load_trace_for_task(failure.get("task", ""), traces_dir)
            future = executor.submit(_analyze_single_failure, llm, failure, trace)
            futures[future] = failure

        for future in as_completed(futures):
            results.append(future.result())

    logger.info("Deep error analysis: %d failures analyzed", len(results))
    return results


def format_error_analysis(analyses: list[dict[str, Any]]) -> str:
    """Format deep error analyses as markdown for the metaprompt."""
    if not analyses:
        return "No deep error analysis available."
    parts: list[str] = []
    for a in analyses:
        parts.append(
            f"### Task: {a['task'][:100]}\n"
            f"- **Root cause:** {a['root_cause']}\n"
            f"- **Should have done:** {a['counterfactual']}\n"
            f"- **Suggested guidance:** {a['suggested_guidance']}"
        )
    return "\n\n".join(parts)
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=. python -m pytest tests/evoagent/integration/test_error_analyzer.py -v
```

- [ ] **Step 4: Update `evoagent/evolution/prompt_optimizer.py`**

Add deep analysis support. Update `DEFAULT_METAPROMPT` to include `{error_analysis}` and update `optimize_prompt`:

```python
DEFAULT_METAPROMPT = (
    "You are a prompt engineering expert.\n\n"
    "## Current Prompt\n{current_prompt}\n\n"
    "## Current Score\n{current_score}\n\n"
    "## Failure Analysis\n{failure_analysis}\n\n"
    "## Common Issues\n{common_issues}\n\n"
    "## Deep Error Analysis\n{error_analysis}\n\n"
    "Generate an improved system prompt that addresses the failures "
    "while preserving what works. Output ONLY the new prompt text."
)


def optimize_prompt(
    llm: BaseChatModel,
    prompt_store: PromptStore,
    analyses: list[dict[str, Any]],
    metaprompt_template: str | None = None,
    traces_dir: Path | None = None,
) -> int:
    """Generate an improved prompt. Returns new version number."""
    version, current_prompt = prompt_store.get_current()

    failed = [a for a in analyses if a.get("classification") in ("failed", "partial")]
    if not failed:
        return version

    issues: list[str] = []
    for a in failed:
        for g in a.get("grader_results", []):
            if isinstance(g, dict) and not g.get("passed", True):
                issues.append(f"[{g.get('name', '?')}] {g.get('reasoning', '')[:150]}")

    # Deep error analysis (parallel)
    error_analysis_text = "No deep error analysis available."
    try:
        from evoagent.evolution.error_analyzer import analyze_failures_deep, format_error_analysis
        deep_results = analyze_failures_deep(llm, failed, traces_dir=traces_dir)
        if deep_results:
            error_analysis_text = format_error_analysis(deep_results)
    except Exception as exc:
        logger.warning("Deep error analysis failed, using shallow: %s", exc)

    template = metaprompt_template or DEFAULT_METAPROMPT
    scores = [a.get("average_score", 0) for a in analyses]
    avg_score = sum(scores) / len(scores) if scores else 0

    metaprompt = template.format(
        current_prompt=current_prompt[:3000],
        current_score=f"{avg_score:.3f}",
        failure_analysis=f"{len(failed)} failed/partial out of {len(analyses)}",
        common_issues="\n".join(f"- {i}" for i in issues[:10]),
        error_analysis=error_analysis_text,
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

- [ ] **Step 5: Update `evoagent/evolution/__init__.py` to export new module**

Add to the imports after the import guard:

```python
from evoagent.evolution.error_analyzer import analyze_failures_deep, format_error_analysis
```

And add to `__all__`.

- [ ] **Step 6: Run all evolution tests**

```bash
PYTHONPATH=. python -m pytest tests/evoagent/integration/test_error_analyzer.py tests/evoagent/ -v
```

- [ ] **Step 7: Commit**

```bash
git add evoagent/evolution/error_analyzer.py evoagent/evolution/prompt_optimizer.py evoagent/evolution/__init__.py tests/evoagent/integration/test_error_analyzer.py
git commit -m "feat(evoagent): add parallel deep error analysis for prompt optimizer

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: ReasoningSandwichMiddleware

**Files:**
- Modify: `evoagent/harness/middleware.py`
- Create: `tests/evoagent/integration/test_reasoning_sandwich.py`

- [ ] **Step 1: Write failing tests**

Create `tests/evoagent/integration/test_reasoning_sandwich.py`:

```python
"""Tests for ReasoningSandwichMiddleware."""

from evoagent.harness.middleware import ReasoningSandwichMiddleware


def test_planning_phase():
    mw = ReasoningSandwichMiddleware(planning_calls=2)
    assert mw.get_reasoning_effort() == "high"  # call 1
    mw.increment_call()
    assert mw.get_reasoning_effort() == "high"  # call 2


def test_implementation_phase():
    mw = ReasoningSandwichMiddleware(planning_calls=2)
    mw._call_count = 2
    assert mw.get_reasoning_effort() == "medium"  # call 3


def test_verification_phase():
    mw = ReasoningSandwichMiddleware(planning_calls=2)
    mw._call_count = 5
    mw._in_verification = True
    assert mw.get_reasoning_effort() == "high"


def test_enter_verification():
    mw = ReasoningSandwichMiddleware(planning_calls=2)
    mw._call_count = 5
    assert mw.get_reasoning_effort() == "medium"
    mw.enter_verification()
    assert mw.get_reasoning_effort() == "high"
```

- [ ] **Step 2: Implement `ReasoningSandwichMiddleware`**

Add to `evoagent/harness/middleware.py`:

```python
# --- Reasoning Sandwich ---

class ReasoningSandwichMiddleware(AgentMiddleware):
    """Allocates reasoning effort across agent phases.

    High reasoning for planning (first N calls) and verification (final phase),
    medium for implementation (middle calls). Only affects models that support
    reasoning_effort params.
    """

    tools: tuple[BaseTool, ...] = ()

    def __init__(
        self,
        planning_effort: str = "high",
        implementation_effort: str = "medium",
        verification_effort: str = "high",
        planning_calls: int = 2,
    ) -> None:
        self._planning_effort = planning_effort
        self._impl_effort = implementation_effort
        self._verif_effort = verification_effort
        self._planning_calls = planning_calls
        self._call_count = 0
        self._in_verification = False

    def increment_call(self) -> None:
        self._call_count += 1

    def enter_verification(self) -> None:
        self._in_verification = True

    def get_reasoning_effort(self) -> str:
        if self._in_verification:
            return self._verif_effort
        if self._call_count < self._planning_calls:
            return self._planning_effort
        return self._impl_effort

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        effort = self.get_reasoning_effort()
        self._call_count += 1
        # Inject reasoning effort hint as metadata if runtime supports it
        logger.debug("ReasoningSandwich: call %d, effort=%s", self._call_count, effort)
        return None
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=. python -m pytest tests/evoagent/integration/test_reasoning_sandwich.py -v
```

- [ ] **Step 4: Commit**

```bash
git add evoagent/harness/middleware.py tests/evoagent/integration/test_reasoning_sandwich.py
git commit -m "feat(evoagent): add ReasoningSandwichMiddleware for phased reasoning effort

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Pre-Completion Task Verification

**Files:**
- Modify: `evoagent/harness/middleware.py`
- Create: `tests/evoagent/integration/test_task_verification.py`

- [ ] **Step 1: Write failing tests**

Create `tests/evoagent/integration/test_task_verification.py`:

```python
"""Tests for pre-completion task verification in SelfVerificationMiddleware."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from evoagent.harness.middleware import verify_output_against_task


def test_verify_addressed_returns_none():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content='{"addressed": true, "missing": []}')
    issues = verify_output_against_task(llm, "Research quantum computing", "# Quantum Report\n..." + "x" * 500)
    assert issues is None


def test_verify_not_addressed_returns_missing():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content='{"addressed": false, "missing": ["No discussion of error correction", "Missing recent papers"]}'
    )
    issues = verify_output_against_task(llm, "Research quantum computing error correction", "# Report\n..." + "x" * 500)
    assert issues is not None
    assert len(issues) >= 1


def test_verify_llm_error_returns_none():
    llm = MagicMock()
    llm.invoke.side_effect = Exception("LLM broke")
    issues = verify_output_against_task(llm, "task", "output " * 200)
    assert issues is None  # fail-open: don't block on verification error
```

- [ ] **Step 2: Implement `verify_output_against_task` and update `SelfVerificationMiddleware`**

Add function to `evoagent/harness/middleware.py`:

```python
TASK_VERIFICATION_PROMPT = (
    "Does this output fully address the task?\n\n"
    "Task: {task}\n\n"
    "Output (preview):\n{output}\n\n"
    'Reply as JSON: {{"addressed": true/false, "missing": ["list of missing aspects"]}}'
)


def verify_output_against_task(
    llm: Any,
    task: str,
    output: str,
) -> list[str] | None:
    """Verify output addresses the task via lightweight LLM call.

    Returns list of missing aspects, or None if fully addressed (or on error).
    """
    try:
        from evoagent.core.parsing import parse_llm_json
        prompt = TASK_VERIFICATION_PROMPT.format(task=task[:500], output=output[:2000])
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = parse_llm_json(response.content)
        if parsed.get("addressed", True):
            return None
        return parsed.get("missing", ["Output does not fully address the task"])
    except Exception as exc:
        logger.debug("Task verification failed (fail-open): %s", exc)
        return None
```

Update `SelfVerificationMiddleware.__init__` to accept `verify_against_task` and `llm`:

```python
def __init__(
    self,
    required_sections: list[str] | None = None,
    error_patterns: list[re.Pattern[str]] | None = None,
    min_length: int = 500,
    max_retries: int = 2,
    verify_against_task: bool = False,
    llm: Any = None,
) -> None:
    self._sections = required_sections or _DEFAULT_REQUIRED_SECTIONS
    self._patterns = error_patterns or _DEFAULT_ERROR_PATTERNS
    self._min_length = min_length
    self._max_retries = max_retries
    self._retry_count = 0
    self._verify_task = verify_against_task
    self._llm = llm
```

Update `after_model` — after structural checks pass, run task verification:

```python
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

    # Structural checks first (free)
    issues = check_output(content, self._sections, self._patterns, self._min_length)

    # Task verification (1 LLM call) only if structural checks pass
    if not issues and self._verify_task and self._llm:
        task_text = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                task_text = getattr(msg, "content", "") or ""
                break
        if task_text:
            missing = verify_output_against_task(self._llm, task_text, content)
            if missing:
                issues.append(f"output does not fully address the task: {'; '.join(missing[:3])}")

    if not issues:
        self._retry_count = 0
        return None

    if self._retry_count >= self._max_retries:
        self._retry_count = 0
        return None

    self._retry_count += 1
    revision = HumanMessage(content=f"SELF-CHECK FAILED: {'; '.join(issues)}. Revise your report.")
    return {"messages": [*messages, revision]}
```

- [ ] **Step 3: Run tests**

```bash
PYTHONPATH=. python -m pytest tests/evoagent/integration/test_task_verification.py -v
```

- [ ] **Step 4: Commit**

```bash
git add evoagent/harness/middleware.py tests/evoagent/integration/test_task_verification.py
git commit -m "feat(evoagent): add pre-completion task verification to SelfVerificationMiddleware

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Update Builder and Exports

**Files:**
- Modify: `evoagent/harness/builder.py`
- Modify: `evoagent/harness/__init__.py`
- Create: `tests/evoagent/integration/test_builder_updated.py`

- [ ] **Step 1: Write failing test**

Create `tests/evoagent/integration/test_builder_updated.py`:

```python
"""Tests for updated default_middleware_stack."""

from evoagent.harness.builder import default_middleware_stack
from evoagent.harness.middleware import (
    ContextAssemblyMiddleware,
    LoopDetectionMiddleware,
    SelfVerificationMiddleware,
    TimeBudgetMiddleware,
    TraceCaptureMiddleware,
    ReasoningSandwichMiddleware,
)


def test_default_stack_has_6_middleware():
    stack = default_middleware_stack()
    assert len(stack) == 6


def test_default_stack_types():
    stack = default_middleware_stack()
    types = [type(m).__name__ for m in stack]
    assert "SelfVerificationMiddleware" in types
    assert "ContextAssemblyMiddleware" in types
    assert "LoopDetectionMiddleware" in types
    assert "TimeBudgetMiddleware" in types
    assert "ReasoningSandwichMiddleware" in types
    assert "TraceCaptureMiddleware" in types


def test_stack_with_env_detection():
    stack = default_middleware_stack(detect_env=True)
    ctx = [m for m in stack if isinstance(m, ContextAssemblyMiddleware)]
    assert len(ctx) == 1
    assert ctx[0]._detect_env is True


def test_stack_with_time_budget():
    stack = default_middleware_stack(budget_seconds=120)
    tb = [m for m in stack if isinstance(m, TimeBudgetMiddleware)]
    assert len(tb) == 1
    assert tb[0]._budget == 120
```

- [ ] **Step 2: Update `evoagent/harness/builder.py`**

```python
"""Factory for building default middleware stacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evoagent.harness.middleware import (
    ContextAssemblyMiddleware,
    LoopDetectionMiddleware,
    ReasoningSandwichMiddleware,
    SelfVerificationMiddleware,
    TimeBudgetMiddleware,
    TraceCaptureMiddleware,
)


def default_middleware_stack(
    task: str = "",
    skills_dir: Path | None = None,
    traces_dir: Path | None = None,
    required_sections: list[str] | None = None,
    detect_env: bool = False,
    working_dir: Path | None = None,
    budget_seconds: int = 300,
    verify_against_task: bool = False,
    llm: Any = None,
) -> list:
    """Build the default harness middleware stack.

    Returns 6 middleware instances in recommended order.
    """
    return [
        SelfVerificationMiddleware(
            required_sections=required_sections,
            verify_against_task=verify_against_task,
            llm=llm,
        ),
        ContextAssemblyMiddleware(
            skills_dir=skills_dir,
            detect_env=detect_env,
            working_dir=working_dir,
        ),
        LoopDetectionMiddleware(),
        TimeBudgetMiddleware(budget_seconds=budget_seconds),
        ReasoningSandwichMiddleware(),
        TraceCaptureMiddleware(task=task, traces_dir=traces_dir),
    ]
```

- [ ] **Step 3: Update `evoagent/harness/__init__.py`**

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
    ReasoningSandwichMiddleware,
    SelfVerificationMiddleware,
    TimeBudgetMiddleware,
    TraceCaptureMiddleware,
)

__all__ = [
    "ContextAssemblyMiddleware",
    "LoopDetectionMiddleware",
    "ReasoningSandwichMiddleware",
    "SelfVerificationMiddleware",
    "TimeBudgetMiddleware",
    "TraceCaptureMiddleware",
    "default_middleware_stack",
]
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. python -m pytest tests/evoagent/integration/test_builder_updated.py -v
```

- [ ] **Step 5: Commit**

```bash
git add evoagent/harness/builder.py evoagent/harness/__init__.py tests/evoagent/integration/test_builder_updated.py
git commit -m "feat(evoagent): update builder and exports with all new middleware

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Memory Contradiction Detection

**Files:**
- Modify: `evoagent/evolution/sleep_review.py`
- Create: `tests/evoagent/integration/test_contradiction_detection.py`

- [ ] **Step 1: Write failing tests**

Create `tests/evoagent/integration/test_contradiction_detection.py`:

```python
"""Tests for memory contradiction detection in sleep review."""

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
    store.store("semantic", "a", {"content": "fact A"})
    store.store("semantic", "b", {"content": "fact A is wrong"})
    llm = MagicMock()
    llm.invoke.side_effect = Exception("LLM broke")
    removed = detect_contradictions(llm, store)
    assert removed == 0  # fail-safe: don't delete on error
```

- [ ] **Step 2: Implement `detect_contradictions` in `evoagent/evolution/sleep_review.py`**

Add the following after the existing `_summarize_trace` function:

```python
CONTRADICTION_PROMPT = (
    "These facts were learned across multiple agent runs. "
    "Do any of them contradict each other?\n\n"
    "Facts:\n{facts}\n\n"
    "Respond as JSON: {\"contradictions\": [{\"fact_a\": \"...\", \"fact_b\": \"...\", "
    "\"resolution\": \"which is correct\", \"keep\": \"key of fact to keep\"}]}\n"
    "If no contradictions, return {\"contradictions\": []}"
)


def detect_contradictions(
    llm: BaseChatModel,
    memory: MemoryBackend,
    max_scan: int = 50,
) -> int:
    """Detect and resolve contradictions in semantic memories.

    Clusters memories by topic similarity, checks each cluster for
    contradictions via LLM, and removes the losing fact.

    Returns count of memories removed.
    """
    items = memory.list_all("semantic")
    if len(items) < 2:
        return 0

    # Only scan regular facts (not meta-instructions)
    scannable: dict[str, dict] = {}
    for key, data in list(items.items())[:max_scan]:
        if data.get("type") != "meta_instruction":
            scannable[key] = data

    if len(scannable) < 2:
        return 0

    # Cluster by similarity
    clusters = _cluster_memories(scannable)

    removed = 0
    for cluster_keys in clusters:
        if len(cluster_keys) < 2:
            continue

        facts_text = "\n".join(
            f"- [{k}] {scannable[k].get('content', str(scannable[k]))}"
            for k in cluster_keys
        )

        try:
            prompt = CONTRADICTION_PROMPT.format(facts=facts_text)
            response = llm.invoke([HumanMessage(content=prompt)])
            parsed = parse_llm_json(response.content)

            for contradiction in parsed.get("contradictions", []):
                keep_key = contradiction.get("keep", "")
                # Remove the other fact(s)
                for k in cluster_keys:
                    if k != keep_key and k in scannable:
                        memory.delete("semantic", k)
                        removed += 1
                        logger.info(
                            "Contradiction resolved: removed '%s', kept '%s'",
                            k, keep_key,
                        )
        except Exception as exc:
            logger.warning("Contradiction detection failed for cluster: %s", exc)

    return removed


def _cluster_memories(
    items: dict[str, dict],
    threshold: float = 0.3,
) -> list[list[str]]:
    """Group memory keys into clusters by content similarity."""
    keys = list(items.keys())
    visited: set[str] = set()
    clusters: list[list[str]] = []

    for i, key_i in enumerate(keys):
        if key_i in visited:
            continue
        cluster = [key_i]
        visited.add(key_i)
        content_i = items[key_i].get("content", str(items[key_i]))

        for j in range(i + 1, len(keys)):
            key_j = keys[j]
            if key_j in visited:
                continue
            content_j = items[key_j].get("content", str(items[key_j]))
            # Reuse jaccard from compression
            words_i = set(content_i.lower().split())
            words_j = set(content_j.lower().split())
            if words_i and words_j:
                similarity = len(words_i & words_j) / len(words_i | words_j)
                if similarity >= threshold:
                    cluster.append(key_j)
                    visited.add(key_j)

        clusters.append(cluster)

    return clusters
```

- [ ] **Step 3: Update `run_sleep_review` to call `detect_contradictions`**

Add the `detect_contradictions` parameter and call:

```python
def run_sleep_review(
    llm: BaseChatModel,
    memory: MemoryBackend,
    traces_dir: Path,
    prompt_template: str | None = None,
    max_traces: int = 20,
    detect_contradictions_flag: bool = True,
) -> dict[str, Any]:
    """Run sleep-time review and store meta-instructions."""
    # Contradiction detection pass (before generating new meta-instructions)
    contradictions_removed = 0
    if detect_contradictions_flag:
        contradictions_removed = detect_contradictions(llm, memory)

    traces = _load_traces(traces_dir, limit=max_traces)
    if not traces:
        return {"meta_instructions": [], "contradictions_removed": contradictions_removed, "status": "no_traces"}

    # ... rest of existing code ...

    return {
        "meta_instructions": result.get("meta_instructions", []),
        "consistent_successes": result.get("consistent_successes", []),
        "recurring_failures": result.get("recurring_failures", []),
        "contradictions_removed": contradictions_removed,
        "status": "complete",
    }
```

- [ ] **Step 4: Run tests**

```bash
PYTHONPATH=. python -m pytest tests/evoagent/integration/test_contradiction_detection.py -v
```

- [ ] **Step 5: Run full test suite**

```bash
PYTHONPATH=. python -m pytest tests/evoagent/ -v
```

- [ ] **Step 6: Commit**

```bash
git add evoagent/evolution/sleep_review.py tests/evoagent/integration/test_contradiction_detection.py
git commit -m "feat(evoagent): add memory contradiction detection to sleep review

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Wire Improvements into src/ Application

**Files:**
- Modify: `src/agent/deep_agent.py`
- Modify: `src/evolution/prompt_optimizer.py`

- [ ] **Step 1: Update `src/agent/deep_agent.py`**

Update the harness middleware stack construction to use new params:

```python
from evoagent.harness.builder import default_middleware_stack

# In create_agent():
harness_middleware = default_middleware_stack(
    task=task,
    skills_dir=settings.skills_path,
    traces_dir=settings.traces_path,
    budget_seconds=settings.task_timeout_seconds if hasattr(settings, 'task_timeout_seconds') else 300,
    detect_env=False,  # research agent doesn't run in sandboxes
)
```

This replaces the manual middleware list construction with the builder.

- [ ] **Step 2: Update `src/evolution/prompt_optimizer.py`**

Add `traces_dir` parameter pass-through to evoagent's `optimize_prompt` if it uses `analyze_failures_deep`. Read the current file and add `traces_dir` to the function signature where `optimize_prompt` from evoagent is called or where failure analysis happens.

- [ ] **Step 3: Verify imports**

```bash
PYTHONPATH=. python -c "
from src.agent.deep_agent import create_agent
from src.evolution.prompt_optimizer import optimize_prompt
from evoagent.harness import default_middleware_stack, TimeBudgetMiddleware, ReasoningSandwichMiddleware
print('All OK')
"
```

- [ ] **Step 4: Run full test suite**

```bash
PYTHONPATH=. python -m pytest tests/evoagent/ tests/unit/ -q
```

All 188+ tests must pass.

- [ ] **Step 5: Commit**

```bash
git add src/agent/deep_agent.py src/evolution/prompt_optimizer.py
git commit -m "feat: wire harness improvements into research agent application

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
