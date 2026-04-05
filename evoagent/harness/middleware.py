"""Harness middleware stack for LangChain-based agents.

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

from langchain.agents.middleware.types import AgentMiddleware
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


class SelfVerificationMiddleware(AgentMiddleware):
    """Checks agent output for completeness before finishing."""

    tools: tuple[BaseTool, ...] = ()

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
        revision = HumanMessage(content=f"SELF-CHECK FAILED: {'; '.join(issues)}. Revise your report.")
        return {"messages": [*messages, revision]}


# --- Context Assembly ---

class ContextAssemblyMiddleware(AgentMiddleware):
    """Enriches first model call with environment context."""

    tools: tuple[BaseTool, ...] = ()

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

    def __init__(self, max_similar: int = 3, max_total: int = 12) -> None:
        self._max_similar = max_similar
        self._max_total = max_total
        self._queries: list[str] = []
        self._warned = False

    def should_warn(self, query: str) -> bool:
        similar_count = sum(1 for q in self._queries if is_similar_query(q, query))
        self._queries.append(query)
        return (similar_count >= self._max_similar or len(self._queries) >= self._max_total) and not self._warned


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
