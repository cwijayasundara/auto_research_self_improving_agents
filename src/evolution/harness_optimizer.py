"""Harness optimizer: reads execution traces and grading results to propose
changes to middleware parameters.

Uses LLM-in-the-loop to analyze failure patterns and suggest HarnessConfig
adjustments, with clamping to keep values within safe bounds.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from evoagent.core.parsing import parse_llm_json
from src.evolution.harness_config import HarnessConfig, HarnessConfigStore
from src.evolution.run_log import RunLogEntry

logger = logging.getLogger(__name__)

# Safe bounds for numeric parameters: param -> (min, max)
_PARAM_BOUNDS: dict[str, tuple[int, int]] = {
    "max_retries": (0, 10),
    "min_length": (100, 5000),
    "max_similar": (1, 20),
    "max_total": (3, 50),
    "max_file_edits": (1, 20),
    "max_repeated_tools": (1, 15),
    "budget_seconds": (60, 600),
    "planning_calls": (1, 5),
    "search_max_retries": (0, 5),
    "search_retry_delay": (0, 10),
    "search_max_results": (1, 10),
}

_VALID_EFFORTS = {"low", "medium", "high"}

HARNESS_OPTIMIZER_PROMPT = """\
You are a middleware parameter optimizer for a research agent.

## Current Configuration
{current_config}

## Diagnosis
{diagnosis}

## Trace Data
{trace_data}

Based on the diagnosis and trace data above, propose changes to the middleware
parameters that would improve the agent's performance. Focus on the dimensions
with the highest failure rates.

Respond with JSON only:
{{
  "changes": {{
    "param_name": new_value,
    ...
  }},
  "reasoning": "Brief explanation of why these changes should help."
}}

Only include parameters you want to change. Valid parameters and their types:
- max_retries (int), min_length (int), verify_against_task (bool)
- required_sections (list[str])
- completion_checks (list[str]): custom quality checks evaluated by LLM
- max_similar (int), max_total (int), max_file_edits (int), max_repeated_tools (int)
- budget_seconds (int), warn_at (list[float])
- planning_effort / implementation_effort / verification_effort: "low" | "medium" | "high"
- planning_calls (int), detect_env (bool)
- search_max_retries (int), search_retry_delay (int), search_max_results (int)
- search_depth: "basic" | "advanced"
"""


def build_harness_diagnosis(
    entries: list[RunLogEntry], current_config: HarnessConfig
) -> str:
    """Build diagnostic text from run log entries and current config.

    Computes per-dimension failure rates and top failure reasons.
    """
    # Collect per-dimension stats
    dim_scores: dict[str, list[float]] = defaultdict(list)
    dim_reasons: dict[str, list[str]] = defaultdict(list)

    for entry in entries:
        for gr in entry.grader_results:
            dim = gr.get("dimension", "unknown")
            score = gr.get("score", 0.0)
            reason = gr.get("reason", "")
            dim_scores[dim].append(score)
            if score < 0.5 and reason:
                dim_reasons[dim].append(reason)

    # Build diagnosis text
    lines: list[str] = []
    lines.append("=== Per-Dimension Failure Analysis ===")
    for dim, scores in sorted(dim_scores.items()):
        avg = sum(scores) / len(scores) if scores else 0.0
        fail_count = sum(1 for s in scores if s < 0.5)
        lines.append(
            f"  {dim}: avg={avg:.3f}, failures={fail_count}/{len(scores)}"
        )
        reasons = dim_reasons.get(dim, [])
        if reasons:
            # Show top 3 most common reasons
            from collections import Counter

            top = Counter(reasons).most_common(3)
            for reason_text, count in top:
                lines.append(f"    - ({count}x) {reason_text}")

    # Overall stats
    avg_scores = [e.average_score for e in entries]
    overall_avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0
    lines.append(f"\n=== Overall: {len(entries)} runs, avg_score={overall_avg:.3f} ===")

    # Current config
    lines.append("\n=== Current Config ===")
    config_dict = current_config.to_dict()
    for key, value in sorted(config_dict.items()):
        lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def _parse_single_proposal(
    llm: Callable[..., str],
    config_dict: dict[str, Any],
    prompt: str,
) -> tuple[HarnessConfig, str, int] | None:
    """Parse a single LLM proposal into a config, reasoning, and change count.

    Returns None if parsing fails entirely.
    """
    from langchain_core.messages import HumanMessage

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_response = response.content
        parsed = parse_llm_json(raw_response)
    except Exception as exc:
        logger.warning("Failed to parse harness proposal: %s", exc)
        return None

    changes = parsed.get("changes", {})
    reasoning = parsed.get("reasoning", "No reasoning provided.")

    # Apply changes to a copy of the current config
    new_dict = dict(config_dict)
    applied_changes = 0
    for key, value in changes.items():
        if key not in config_dict:
            logger.warning("Ignoring unknown parameter: %s", key)
            continue

        # Validate effort strings
        if key in ("planning_effort", "implementation_effort", "verification_effort"):
            if value not in _VALID_EFFORTS:
                logger.warning(
                    "Ignoring invalid effort value %r for %s", value, key
                )
                continue

        # Validate search_depth
        if key == "search_depth":
            if value not in ("basic", "advanced"):
                logger.warning(
                    "Ignoring invalid search_depth value %r", value
                )
                continue

        # Clamp numeric values
        if key in _PARAM_BOUNDS:
            lo, hi = _PARAM_BOUNDS[key]
            value = max(lo, min(hi, int(value)))

        if new_dict[key] != value:
            applied_changes += 1
        new_dict[key] = value

    new_config = HarnessConfig.from_dict(new_dict)
    return new_config, reasoning, applied_changes


def propose_harness_changes(
    llm: Callable[..., str],
    diagnosis: str,
    current_config: HarnessConfig,
    trace_data: str = "",
    n_candidates: int = 2,
) -> tuple[HarnessConfig, str]:
    """Use LLM to propose parameter changes based on diagnosis.

    Generates n_candidates proposals and picks the most conservative one
    (fewest parameter changes from current config) to prevent over-tuning.

    Returns (new_config, reasoning_string).
    """
    config_dict = current_config.to_dict()
    prompt = HARNESS_OPTIMIZER_PROMPT.format(
        current_config=json.dumps(config_dict, indent=2),
        diagnosis=diagnosis,
        trace_data=trace_data or "(no trace data available)",
    )

    # Generate multiple candidates and pick the most conservative
    candidates: list[tuple[HarnessConfig, str, int]] = []
    for i in range(n_candidates):
        result = _parse_single_proposal(llm, config_dict, prompt)
        if result is not None:
            candidates.append(result)

    if not candidates:
        logger.warning("All harness proposals failed, returning current config")
        return current_config, "No valid proposals generated."

    # Pick the most conservative candidate (fewest changes from current)
    best = min(candidates, key=lambda c: c[2])
    logger.info(
        "Picked most conservative harness proposal (%d changes) from %d candidates",
        best[2],
        len(candidates),
    )
    return best[0], best[1]


def _load_trace(trace_path: str) -> str:
    """Load a trace JSON file and return a compact summary."""
    if not trace_path:
        return ""
    path = Path(trace_path)
    if not path.exists():
        return ""
    try:
        with open(path) as f:
            data = json.load(f)

        lines: list[str] = []
        lines.append(f"Trace: {path.name}")

        # Duration
        if "duration_seconds" in data:
            lines.append(f"  Duration: {data['duration_seconds']:.1f}s")

        # Steps
        steps = data.get("steps", [])
        lines.append(f"  Steps: {len(steps)}")

        # Tool calls summary
        tool_calls: list[str] = []
        errors: list[str] = []
        for step in steps:
            tool = step.get("tool", step.get("action", ""))
            args = step.get("args", step.get("input", ""))
            if tool:
                summary = f"{tool}"
                if args:
                    arg_str = str(args)[:80]
                    summary += f"({arg_str})"
                tool_calls.append(summary)
            error = step.get("error", "")
            if error:
                errors.append(str(error)[:120])

        if tool_calls:
            lines.append(f"  Tool calls: {', '.join(tool_calls[:10])}")
        if errors:
            lines.append(f"  Errors: {'; '.join(errors[:5])}")

        return "\n".join(lines)
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Failed to load trace %s: %s", trace_path, exc)
        return ""


def optimize_harness(
    llm: Callable[..., str],
    entries: list[RunLogEntry],
    config_store: HarnessConfigStore,
    trace_fetcher=None,
) -> int:
    """Main entry point: diagnose, collect traces, propose changes, save.

    Uses LangSmith traces when available for richer diagnostics,
    falls back to local trace files.

    Returns the new config version number.
    """
    current_config = config_store.load_best()

    # Build diagnosis
    diagnosis = build_harness_diagnosis(entries, current_config)

    # Collect trace data — prefer LangSmith, fall back to local
    trace_data = ""
    if trace_fetcher is not None:
        trace_data = trace_fetcher.fetch_traces_for_entries(entries, max_traces=5)

    if not trace_data:
        trace_parts: list[str] = []
        for entry in entries[:5]:
            tp = getattr(entry, "trace_path", "")
            if tp:
                summary = _load_trace(tp)
                if summary:
                    trace_parts.append(summary)
        trace_data = "\n\n".join(trace_parts)

    # Propose changes
    new_config, reasoning = propose_harness_changes(
        llm, diagnosis, current_config, trace_data
    )

    # Only save if something changed
    if new_config.to_dict() == current_config.to_dict():
        logger.info("Harness optimizer: no changes proposed")
        return config_store.get_latest_version()

    # Compute average score from entries for the new config
    avg_score = (
        sum(e.average_score for e in entries) / len(entries) if entries else None
    )

    version = config_store.save(new_config, score=avg_score)
    logger.info(
        "Harness optimizer saved version %d: %s", version, reasoning[:100]
    )
    return version
