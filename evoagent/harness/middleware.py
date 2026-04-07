"""Harness middleware stack for LangChain-based agents.

1. SelfVerificationMiddleware — catches incomplete outputs
2. ContextAssemblyMiddleware — enriches first model call
3. LoopDetectionMiddleware — detects repetitive searches
4. TraceCaptureMiddleware — records traces for offline analysis
5. TimeBudgetMiddleware — injects time pressure warnings into tool responses
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# --- Self-Verification ---

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


TASK_VERIFICATION_PROMPT = (
    "Does this output fully address the task?\n\n"
    "Task: {task}\n\n"
    "Output (preview):\n{output}\n\n"
    'Reply as JSON: {{"addressed": true, "missing": []}} or {{"addressed": false, "missing": ["list of missing aspects"]}}'
)


COMPLETION_CHECK_PROMPT = (
    "Evaluate if this output passes these quality checks.\n\n"
    "Task: {task}\n\n"
    "Output (preview):\n{output}\n\n"
    "Checks:\n{checks}\n\n"
    'Reply as JSON: {{"passed": true}} or {{"passed": false, "failures": ["list of failed checks"]}}'
)


def _evaluate_completion_checks(llm: Any, task: str, output: str, checks: list[str]) -> list[str]:
    """Evaluate custom completion checks via LLM. Returns list of failures."""
    from evoagent.core.parsing import parse_llm_json
    checks_text = "\n".join(f"- {c}" for c in checks)
    prompt = COMPLETION_CHECK_PROMPT.format(
        task=task[:500],
        output=output[:2000],
        checks=checks_text,
    )
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = parse_llm_json(response.content)
        if parsed and not parsed.get("passed", True):
            return parsed.get("failures", ["Custom check failed"])
    except Exception:
        pass
    return []


def verify_output_against_task(llm: Any, task: str, output: str) -> list[str] | None:
    """Verify output addresses the task via lightweight LLM call.
    Returns list of missing aspects, or None if addressed (or on error — fail-open).
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


class SelfVerificationMiddleware(AgentMiddleware):
    """Checks agent output for completeness before finishing."""

    tools: tuple[BaseTool, ...] = ()

    def __init__(
        self,
        required_sections: list[str] | None = None,
        error_patterns: list[re.Pattern[str]] | None = None,
        min_length: int = 500,
        max_retries: int = 2,
        verify_against_task: bool = False,
        llm: Any = None,
        completion_checks: list[str] | None = None,
    ) -> None:
        self._sections = required_sections or _DEFAULT_REQUIRED_SECTIONS
        self._patterns = error_patterns or _DEFAULT_ERROR_PATTERNS
        self._min_length = min_length
        self._max_retries = max_retries
        self._retry_count = 0
        self._verify_task = verify_against_task
        self._llm = llm
        self._completion_checks = completion_checks or []

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: Any, runtime: Any) -> dict[str, Any] | None:
        """Check the just-emitted AIMessage and request a revision if it fails.

        To actually trigger the revision, this hook must:
        1. Be decorated with @hook_config(can_jump_to=["model"]) so the graph
           wires a conditional edge from this node back to "model".
        2. Return ``{"jump_to": "model"}`` in addition to the new messages.

        Without (1) and (2), LangChain's routing function (see
        ``_make_model_to_tools_edge`` in ``langchain.agents.factory``) walks
        backwards to find the last AIMessage, sees its empty ``tool_calls``,
        and exits to END — silently dropping any HumanMessage we appended.
        That bug caused max_retries to be dead config and the SELF-CHECK
        revision pass to never run.
        """
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

        if not issues and self._verify_task and self._llm:
            task_text = ""
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    task_text = getattr(msg, "content", "") or ""
                    break
            if task_text:
                missing = verify_output_against_task(self._llm, task_text, content)
                if missing:
                    issues.extend(missing)

        # Custom completion checks (LLM-evaluated)
        if not issues and self._completion_checks and self._llm:
            task_text = ""
            for msg in messages:
                if hasattr(msg, "type") and msg.type == "human":
                    task_text = (msg.content or "")[:500]
                    break
            if task_text:
                check_issues = _evaluate_completion_checks(
                    self._llm, task_text, content, self._completion_checks
                )
                if check_issues:
                    issues.extend(check_issues)

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
        # Return only the NEW message (the add_messages reducer will append it)
        # and explicitly jump back to the model node so the revision actually
        # runs. The matching @hook_config decorator above is what makes the
        # conditional edge available.
        return {"messages": [revision], "jump_to": "model"}


# --- Context Assembly ---

_TOOLS_TO_DETECT = ["python3", "python", "node", "npm", "curl", "git", "make", "gcc", "java", "go"]


def detect_environment(
    working_dir: str | Path,
    max_entries: int = 50,
    max_depth: int = 2,
) -> dict[str, Any]:
    """Detect the current environment and return a structured dict.

    Returns:
        working_directory: str path of the working directory
        directory_listing: str with entries (dirs get '/' suffix, 2-level deep,
            capped at max_entries, skips dotfiles)
        available_tools: list[str] of tool names found via shutil.which()
    """
    working_dir = Path(working_dir)

    # Build directory listing (BFS up to max_depth, skip dotfiles, cap at max_entries)
    entries: list[str] = []

    def _collect(directory: Path, depth: int) -> None:
        if depth > max_depth or len(entries) >= max_entries:
            return
        try:
            items = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return
        for item in items:
            if len(entries) >= max_entries:
                break
            if item.name.startswith("."):
                continue
            rel = item.relative_to(working_dir)
            entries.append(str(rel) + ("/" if item.is_dir() else ""))
            if item.is_dir() and depth < max_depth:
                _collect(item, depth + 1)

    _collect(working_dir, depth=1)

    directory_listing = "\n".join(entries) if entries else "(empty)"

    # Detect available tools
    available_tools = [tool for tool in _TOOLS_TO_DETECT if shutil.which(tool) is not None]

    return {
        "working_directory": str(working_dir),
        "directory_listing": directory_listing,
        "available_tools": available_tools,
    }


def format_environment_context(env: dict[str, Any]) -> str:
    """Format the environment dict as a markdown block."""
    tools_str = ", ".join(env.get("available_tools", [])) or "none detected"
    return (
        "## System Environment\n"
        f"**Working directory:** `{env.get('working_directory', '')}`\n\n"
        f"**Available tools:** {tools_str}\n\n"
        f"**Directory listing:**\n```\n{env.get('directory_listing', '')}\n```"
    )


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

        if self._detect_env:
            try:
                wd = self._working_dir or Path(os.getcwd())
                env = detect_environment(wd)
                context_parts.append(format_environment_context(env))
            except Exception:
                pass

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


# --- Loop Detection ---

def is_similar_query(q1: str, q2: str) -> bool:
    """Check if two queries are substantially similar (>60% word overlap)."""
    words1 = set(re.sub(r"[^\w\s]", "", q1.lower()).split())
    words2 = set(re.sub(r"[^\w\s]", "", q2.lower()).split())
    if not words1 or not words2:
        return False
    return len(words1 & words2) / min(len(words1), len(words2)) > 0.6


class LoopDetectionMiddleware(AgentMiddleware):
    """Detects repetitive search queries and nudges toward synthesis."""

    tools: tuple[BaseTool, ...] = ()

    _FILE_EDIT_TOOLS = {"write_file", "edit_file", "create_file", "patch"}

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
        self._tool_calls: list[tuple[str, str]] = []
        self._tool_warned: bool = False

    def should_warn(self, query: str) -> bool:
        similar_count = sum(1 for q in self._queries if is_similar_query(q, query))
        self._queries.append(query)
        return (similar_count >= self._max_similar or len(self._queries) >= self._max_total) and not self._warned

    def should_warn_file_edit(self, file_path: str) -> bool:
        """Increment edit count for file_path, return True if threshold reached and not yet warned."""
        self._file_edit_counts[file_path] = self._file_edit_counts.get(file_path, 0) + 1
        if self._file_edit_counts[file_path] >= self._max_file_edits and file_path not in self._file_edit_warned:
            self._file_edit_warned.add(file_path)
            return True
        return False

    def should_warn_repeated_tool(self, tool_name: str, args_str: str) -> bool:
        """Track tool calls, return True if >= max_repeated_tools similar calls exist and not yet warned."""
        self._tool_calls.append((tool_name, args_str))
        if self._tool_warned:
            return False
        similar_count = sum(
            1
            for name, args in self._tool_calls
            if name == tool_name and is_similar_query(args, args_str)
        )
        if similar_count >= self._max_repeated_tools:
            self._tool_warned = True
            return True
        return False

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        """Call handler then append loop-detection warnings if thresholds are crossed."""
        result = handler(request)

        tool_name: str = getattr(request, "name", "") or getattr(request, "tool_name", "") or ""
        tool_args: dict[str, Any] = {}
        raw_args = getattr(request, "args", None) or getattr(request, "arguments", None) or {}
        if isinstance(raw_args, dict):
            tool_args = raw_args

        # Determine file path from common arg key names
        file_path: str | None = None
        for key in ("file_path", "path", "filename"):
            val = tool_args.get(key)
            if isinstance(val, str) and val:
                file_path = val
                break

        warnings: list[str] = []

        # File-edit tracking
        if tool_name in self._FILE_EDIT_TOOLS and file_path:
            if self.should_warn_file_edit(file_path):
                warnings.append(
                    f"LOOP WARNING: '{file_path}' has been edited {self._max_file_edits}+ times. "
                    "Consider whether further edits are necessary."
                )

        # Repeated tool tracking
        args_str = str(tool_args)
        if self.should_warn_repeated_tool(tool_name, args_str):
            warnings.append(
                f"LOOP WARNING: tool '{tool_name}' has been called {self._max_repeated_tools}+ times "
                "with similar arguments. Consider synthesizing results instead of repeating."
            )

        if not warnings:
            return result

        content = getattr(result, "content", None)
        if not isinstance(content, str):
            return result

        from langchain_core.messages import ToolMessage  # noqa: PLC0415

        tool_call_id = getattr(result, "tool_call_id", "") or ""
        warning_text = "\n".join(warnings)
        warning_msg = ToolMessage(content=warning_text, tool_call_id=tool_call_id)
        result.__dict__.setdefault("_loop_detection_warnings", []).append(warning_msg)
        return result


# --- Trace Capture ---

class TraceCaptureMiddleware(AgentMiddleware):
    """Records execution traces to disk for offline analysis."""

    tools: tuple[BaseTool, ...] = ()
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


# --- Time Budget ---

class TimeBudgetMiddleware(AgentMiddleware):
    """Injects time-pressure warnings into tool responses as the budget is consumed."""

    tools: tuple[BaseTool, ...] = ()

    _WARN_MESSAGES: dict[str, str] = {
        "urgent": "URGENT: stop all searches, write report NOW",
        "normal": "Start synthesizing",
    }

    def __init__(
        self,
        budget_seconds: int = 300,
        warn_at: list[float] | None = None,
    ) -> None:
        self._budget = budget_seconds
        self._warn_at: list[float] = warn_at if warn_at is not None else [0.6, 0.85]
        self._start_time: float = 0.0
        self._warnings_fired: set[float] = set()

    def start(self) -> None:
        """Record the start time."""
        self._start_time = time.monotonic()

    def before_agent(self, state: Any, runtime: Any) -> None:
        """Auto-start timer if not already started."""
        if self._start_time == 0.0:
            self.start()
        return None

    def elapsed_fraction(self) -> float:
        """Return fraction of budget consumed. Returns 0.0 if not started."""
        if self._start_time == 0.0:
            return 0.0
        return (time.monotonic() - self._start_time) / self._budget

    def get_warning(self) -> str | None:
        """Check thresholds and return a warning string if one should fire.

        Thresholds are checked in descending order so the most urgent message
        wins when multiple thresholds are crossed simultaneously.
        """
        fraction = self.elapsed_fraction()
        for threshold in sorted(self._warn_at, reverse=True):
            if fraction >= threshold and threshold not in self._warnings_fired:
                self._warnings_fired.add(threshold)
                if threshold >= 0.85:
                    return self._WARN_MESSAGES["urgent"]
                return self._WARN_MESSAGES["normal"]
        return None

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        """Call handler then append a time-pressure warning if one is due."""
        result = handler(request)
        warning = self.get_warning()
        if warning is None:
            return result

        # Only append when the result carries string content.
        content = getattr(result, "content", None)
        if not isinstance(content, str):
            return result

        from langchain_core.messages import ToolMessage  # noqa: PLC0415

        tool_call_id = getattr(result, "tool_call_id", "") or ""
        warning_msg = ToolMessage(content=warning, tool_call_id=tool_call_id)
        # Return the original result; attach warning as a sibling attribute so
        # callers that inspect the object can find it without breaking the
        # standard ToolMessage interface.
        result.__dict__.setdefault("_time_budget_warning", warning_msg)
        return result


# --- Reasoning Sandwich ---

class ReasoningSandwichMiddleware(AgentMiddleware):
    """Allocates reasoning effort across agent phases."""

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
        logger.debug("ReasoningSandwich: call %d, effort=%s", self._call_count, effort)
        return None
