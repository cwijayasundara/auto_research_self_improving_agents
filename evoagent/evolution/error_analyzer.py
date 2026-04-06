"""Parallel deep error analysis for failed research analyses."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evoagent.core.parsing import parse_llm_json

logger = logging.getLogger(__name__)

ERROR_ANALYSIS_PROMPT = """\
You are a diagnostic expert analyzing a failed AI research analysis.

## Task
{task}

## Agent Output
{output}

## Score
{score}

## Grader Feedback
{grader_feedback}

## Execution Trace
{trace}

Analyze this failure and respond with JSON only (no markdown fences):
{{
  "root_cause": "<concise description of the primary failure cause>",
  "counterfactual": "<what the agent should have done differently>",
  "suggested_guidance": "<specific instruction to add to the system prompt to prevent this failure>"
}}
"""


def _load_trace_for_task(task: str, traces_dir: Path | None) -> str:
    """Find and load a matching execution trace file by task prefix."""
    if traces_dir is None:
        return ""
    if not traces_dir.is_dir():
        return ""

    # Normalize task to a prefix for filename matching
    prefix = task[:40].lower().replace(" ", "_").replace("/", "_")
    for path in traces_dir.iterdir():
        if path.is_file() and prefix in path.name.lower():
            try:
                return path.read_text(encoding="utf-8")[:2000]
            except OSError:
                return ""
    return ""


def _analyze_single_failure(
    llm: BaseChatModel,
    failure: dict[str, Any],
    trace: str = "",
) -> dict[str, Any]:
    """Run a single LLM call to analyze one failure."""
    task = failure.get("task", "")
    output = failure.get("output", "")
    score = failure.get("average_score", 0.0)
    grader_results = failure.get("grader_results", [])

    grader_lines: list[str] = []
    for g in grader_results:
        if isinstance(g, dict):
            name = g.get("name", "?")
            passed = g.get("passed", True)
            reasoning = g.get("reasoning", "")[:200]
            grader_lines.append(f"[{name}] passed={passed}: {reasoning}")

    prompt = ERROR_ANALYSIS_PROMPT.format(
        task=str(task)[:500],
        output=str(output)[:1000],
        score=f"{score:.3f}",
        grader_feedback="\n".join(grader_lines) or "No grader feedback available.",
        trace=trace[:1500] if trace else "No trace available.",
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = parse_llm_json(response.content)
        return {
            "task": task,
            "root_cause": parsed.get("root_cause", "unknown"),
            "counterfactual": parsed.get("counterfactual", ""),
            "suggested_guidance": parsed.get("suggested_guidance", ""),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Error analyzing failure for task %r: %s", str(task)[:60], exc)
        return {
            "task": task,
            "root_cause": "analysis_failed",
            "counterfactual": "",
            "suggested_guidance": "",
        }


def analyze_failures_deep(
    llm: BaseChatModel,
    failed_analyses: list[dict[str, Any]],
    traces_dir: Path | None = None,
    max_parallel: int = 3,
    max_failures: int = 5,
) -> list[dict[str, Any]]:
    """Analyze failures in parallel using an LLM.

    Args:
        llm: Chat model to use for analysis.
        failed_analyses: List of failed analysis dicts, each with keys:
            task, output, average_score, grader_results.
        traces_dir: Optional directory containing execution trace files.
        max_parallel: Maximum number of concurrent LLM calls.
        max_failures: Maximum number of failures to analyze.

    Returns:
        List of dicts with keys: task, root_cause, counterfactual, suggested_guidance.
    """
    subset = failed_analyses[:max_failures]
    if not subset:
        return []

    results: list[dict[str, Any]] = [{}] * len(subset)

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_idx = {}
        for idx, failure in enumerate(subset):
            trace = _load_trace_for_task(failure.get("task", ""), traces_dir)
            future = executor.submit(_analyze_single_failure, llm, failure, trace)
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Unexpected error in error analysis future: %s", exc)
                results[idx] = {
                    "task": subset[idx].get("task", ""),
                    "root_cause": "analysis_failed",
                    "counterfactual": "",
                    "suggested_guidance": "",
                }

    return results


def format_error_analysis(analyses: list[dict[str, Any]]) -> str:
    """Format a list of error analysis dicts as markdown for the metaprompt."""
    if not analyses:
        return "No error analysis available."

    lines: list[str] = []
    for i, analysis in enumerate(analyses, start=1):
        task = str(analysis.get("task", ""))[:80]
        root_cause = analysis.get("root_cause", "unknown")
        counterfactual = analysis.get("counterfactual", "")
        guidance = analysis.get("suggested_guidance", "")

        lines.append(f"### Failure {i}: {task}")
        lines.append(f"- **Root Cause:** {root_cause}")
        if counterfactual:
            lines.append(f"- **Counterfactual:** {counterfactual}")
        if guidance:
            lines.append(f"- **Suggested Guidance:** {guidance}")
        lines.append("")

    return "\n".join(lines).strip()
