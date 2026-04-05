"""Sleep-time compute: offline cross-run trace analysis."""

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
