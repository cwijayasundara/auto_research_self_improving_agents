"""Prompt optimizer using metaprompt approach.

Enhanced with skill-awareness: the optimizer can reference learned skills
when generating improved prompts, creating tighter integration between
the skill learning and prompt optimization loops.

Includes validation guardrails to prevent prompt drift — e.g. the optimizer
generating prompts that ask the user for input, breaking autonomous operation.
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agent.prompt_store import PromptStore
from src.agent.prompts import METAPROMPT_TEMPLATE
from src.evolution.state import AnalysisResult
from evoagent.skills.manager import SkillManager

TRACES_DIR = Path("traces")

logger = logging.getLogger(__name__)

# Patterns that indicate the prompt drifted into asking for user interaction.
# Each tuple is (compiled regex, human-readable description).
_AUTONOMY_VIOLATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ask\s+(the\s+)?user", re.IGNORECASE), "asks the user"),
    (re.compile(r"would you like", re.IGNORECASE), "asks 'would you like'"),
    (re.compile(r"choose\s+(one|an option|from|between|\d)", re.IGNORECASE), "presents choices"),
    (re.compile(r"select\s+(one|an option|from|\d)", re.IGNORECASE), "asks to select"),
    (re.compile(r"(?:option|choice)\s*[123]\)?", re.IGNORECASE), "presents numbered options"),
    (
        re.compile(r"please\s+(choose|select|pick|specify|clarify|confirm)", re.IGNORECASE),
        "requests user action",
    ),
    (re.compile(r"do you (?:want|prefer|need)", re.IGNORECASE), "asks user preference"),
    (re.compile(r"let me know (?:if|which|what|how)", re.IGNORECASE), "requests user feedback"),
    (re.compile(r"waiting for.*(?:input|response|reply)", re.IGNORECASE), "waits for input"),
]


def validate_prompt_autonomy(prompt: str) -> list[str]:
    """Check a generated prompt for patterns that violate autonomous operation.

    Returns a list of violation descriptions. Empty list means the prompt is safe.
    """
    violations: list[str] = []
    for pattern, description in _AUTONOMY_VIOLATION_PATTERNS:
        if pattern.search(prompt):
            violations.append(description)
    return violations


def _build_trace_digest(task: str, traces_dir: Path | None = None) -> str:
    """Build a trace digest for a failed task by reading its trace file.

    Matches traces to tasks by task text since the orchestrator generates
    run_ids after the agent runs (and orchestrator is immutable).
    """
    search_dir = traces_dir or TRACES_DIR
    if not search_dir.exists():
        return ""

    # Find the most recent trace matching this task
    best_trace: dict[str, Any] | None = None
    best_time = 0.0
    for trace_file in search_dir.glob("*.json"):
        try:
            data = json.loads(trace_file.read_text())
            if data.get("task", "")[:100] == task[:100]:
                ts = trace_file.stat().st_mtime
                if ts > best_time:
                    best_time = ts
                    best_trace = data
        except Exception:
            continue

    if not best_trace:
        return ""

    # Extract key diagnostic info from the trace
    parts: list[str] = []
    parts.append(f"Duration: {best_trace.get('duration_seconds', '?')}s")
    parts.append(f"Total steps: {best_trace.get('step_count', '?')}")

    search_queries: list[str] = []
    errors: list[str] = []
    for step in best_trace.get("steps", []):
        if step.get("type") == "tool_call":
            args = step.get("args_preview", "")
            if "query" in args:
                search_queries.append(args[:100])
            output = step.get("output_preview", "")
            if any(kw in output.lower() for kw in ["error", "fail", "quota", "timeout"]):
                errors.append(output[:150])
        elif step.get("type") == "model_call":
            tool_calls = step.get("tool_calls", [])
            if tool_calls:
                parts.append(f"Model called tools: {[tc['name'] for tc in tool_calls]}")

    if search_queries:
        parts.append(
            f"Search queries tried ({len(search_queries)}): " + "; ".join(search_queries[:5])
        )
    if errors:
        parts.append(f"Errors encountered ({len(errors)}): " + "; ".join(errors[:3]))

    return "\n".join(parts)


def _build_dimension_breakdown(analyses: list[AnalysisResult]) -> str:
    """Build a per-dimension score summary across all trajectories.

    Helps the prompt optimizer see which grading dimension is the bottleneck
    rather than just seeing an opaque average score.
    """
    from collections import defaultdict

    dim_scores: dict[str, list[float]] = defaultdict(list)
    dim_fails: dict[str, int] = defaultdict(int)

    for analysis in analyses:
        for grader in analysis["grader_results"]:
            dim_scores[grader.name].append(grader.score)
            if not grader.passed:
                dim_fails[grader.name] += 1

    if not dim_scores:
        return ""

    total = len(analyses)
    lines = ["Dimension | Avg Score | Fail Rate | Status"]
    lines.append("--- | --- | --- | ---")
    for name in sorted(dim_scores.keys()):
        scores = dim_scores[name]
        avg = sum(scores) / len(scores)
        fail_rate = dim_fails[name] / total
        status = "BOTTLENECK" if fail_rate > 0.5 else "OK" if fail_rate == 0 else "WEAK"
        lines.append(f"{name} | {avg:.2f} | {fail_rate:.0%} | {status}")

    return "\n".join(lines)


def analyze_failures(
    analyses: list[AnalysisResult],
    traces_dir: Path | None = None,
    trace_fetcher: Any | None = None,
) -> dict[str, Any]:
    """Aggregate failure patterns from failed/partial trajectories.

    Enhanced with trace digest and per-dimension breakdown to give the
    prompt optimizer actionable diagnostic context. Uses LangSmith traces
    when available for richer diagnostics.
    """
    failed = [a for a in analyses if a["classification"] in ("failed", "partial")]
    if not failed:
        return {
            "failure_analysis": "No failures to analyze.",
            "common_issues": [],
            "trace_digest": "",
            "dimension_breakdown": "",
        }

    issues: list[str] = []
    for analysis in failed:
        for grader in analysis["grader_results"]:
            if not grader.passed:
                issues.append(
                    f"[{grader.name}] {grader.reasoning} (task: {analysis['task'][:80]})"
                )

    unique_issues = list(dict.fromkeys(issues))

    # Build trace digests for failed trajectories (max 3)
    # Try LangSmith first for rich traces, fall back to local
    trace_digests: list[str] = []
    if trace_fetcher is not None:
        for analysis in failed[:3]:
            rich = trace_fetcher.fetch_rich_trace_for_task(analysis["task"], max_results=1)
            if rich:
                trace_digests.append(f"### Trace for: {analysis['task'][:80]}\n{rich}")
    if not trace_digests:
        for analysis in failed[:3]:
            digest = _build_trace_digest(analysis["task"], traces_dir)
            if digest:
                trace_digests.append(f"### Trace for: {analysis['task'][:80]}\n{digest}")

    failure_summary = (
        f"Analyzed {len(failed)} failed/partial trajectories. "
        f"Found {len(unique_issues)} distinct issues."
    )

    return {
        "failure_analysis": failure_summary,
        "common_issues": unique_issues[:10],
        "trace_digest": "\n\n".join(trace_digests) if trace_digests else "",
        "dimension_breakdown": _build_dimension_breakdown(analyses),
    }


MAX_AUTONOMY_RETRIES = 2


def _generate_single_candidate(
    llm: BaseChatModel,
    base_meta: str,
    current_prompt: str,
) -> str | None:
    """Generate a single candidate prompt with autonomy validation retries.

    Returns the valid prompt string, or None if all retries fail.
    """
    meta = base_meta
    improved = ""
    for attempt in range(1 + MAX_AUTONOMY_RETRIES):
        response = llm.invoke([HumanMessage(content=meta)])
        improved = response.content.strip()

        if "{memory_context}" not in improved:
            improved += "\n{memory_context}"

        violations = validate_prompt_autonomy(improved)
        if not violations:
            return improved

        logger.warning(
            "Prompt autonomy violation (attempt %d/%d): %s",
            attempt + 1,
            1 + MAX_AUTONOMY_RETRIES,
            ", ".join(violations),
        )

        # Inject specific violation feedback for the next attempt
        violation_list = ", ".join(f'"{v}"' for v in violations)
        meta = (
            base_meta
            + f"\n\n## VIOLATION FEEDBACK FROM PREVIOUS ATTEMPT\n"
            f"Your previous output was REJECTED because it contained these "
            f"forbidden patterns: {violation_list}.\n"
            f"These phrases MUST NOT appear anywhere in the prompt — not even "
            f"in a negation like 'do not ask the user'. Instead of referencing "
            f"user interaction at all, simply instruct the agent to act "
            f"autonomously and produce a complete report.\n"
            f"Generate the prompt again WITHOUT any of these patterns."
        )

    # All retries failed — try stripping violations as last resort
    cleaned = _strip_autonomy_violations(improved)
    remaining = validate_prompt_autonomy(cleaned)
    if not remaining:
        logger.info(
            "Prompt autonomy violations stripped automatically after %d failed attempts.",
            1 + MAX_AUTONOMY_RETRIES,
        )
        return cleaned

    return None


def generate_improved_prompt(
    llm: BaseChatModel,
    current_prompt: str,
    current_score: float,
    failure_info: dict[str, Any],
    skills_summary: str = "",
    n_candidates: int = 2,
) -> str:
    """Generate an improved prompt using the metaprompt approach.

    Generates n_candidates prompts (each with autonomy validation retries)
    and picks the one whose length is closest to the current prompt to
    prevent prompt bloat or excessive drift. This selection is zero-cost
    (no extra LLM calls).

    Enhanced: includes a summary of available skills so the prompt
    can reference them for better integration.
    """
    base_meta = METAPROMPT_TEMPLATE.format(
        current_prompt=current_prompt,
        current_score=f"{current_score:.3f}",
        failure_analysis=failure_info["failure_analysis"],
        common_issues="\n".join(f"- {i}" for i in failure_info["common_issues"]),
        available_skills=skills_summary or "No skills learned yet.",
        trace_digest=failure_info.get("trace_digest", "No trace data available."),
        dimension_breakdown=failure_info.get("dimension_breakdown", ""),
    )

    # Generate multiple candidates
    candidates: list[str] = []
    for i in range(n_candidates):
        candidate = _generate_single_candidate(llm, base_meta, current_prompt)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        logger.error(
            "All %d prompt candidates failed autonomy validation. "
            "Keeping current prompt to prevent drift.",
            n_candidates,
        )
        return current_prompt

    if len(candidates) == 1:
        return candidates[0]

    # Pick the candidate with least length drift from current prompt
    current_len = len(current_prompt)
    best = min(candidates, key=lambda c: abs(len(c) - current_len))
    logger.info(
        "Picked prompt candidate with least drift (%+d chars) from %d candidates",
        len(best) - current_len,
        len(candidates),
    )
    return best


def _strip_autonomy_violations(prompt: str) -> str:
    """Remove lines containing autonomy violation patterns as a last resort."""
    lines = prompt.split("\n")
    clean_lines: list[str] = []
    for line in lines:
        has_violation = False
        for pattern, _ in _AUTONOMY_VIOLATION_PATTERNS:
            if pattern.search(line):
                has_violation = True
                break
        if not has_violation:
            clean_lines.append(line)
    return "\n".join(clean_lines)


def _select_tasks_for_mini_scoring(
    analyses: list[AnalysisResult],
    max_tasks: int = 2,
) -> list[AnalysisResult]:
    """Select tasks for mini-scoring, prioritizing failures."""
    if not analyses:
        return []
    failed = [a for a in analyses if a["classification"] in ("failed", "partial")]
    failed.sort(key=lambda a: a["average_score"])
    selected = failed[:max_tasks]
    if len(selected) < max_tasks:
        successful = [a for a in analyses if a["classification"] == "successful"]
        successful.sort(key=lambda a: a["average_score"])
        selected.extend(successful[: max_tasks - len(selected)])
    return selected


def pairwise_compare(
    llm: BaseChatModel,
    task: str,
    output_a: str,
    output_b: str,
    label_a: str,
    label_b: str,
) -> dict[str, str]:
    """Compare two outputs. Randomly swaps A/B to counter position bias."""
    from src.agent.prompts import PAIRWISE_COMPARISON_PROMPT

    if random.random() < 0.5:
        pos_a_out, pos_b_out = output_a[:3000], output_b[:3000]
        pos_a_lbl, pos_b_lbl = label_a, label_b
    else:
        pos_a_out, pos_b_out = output_b[:3000], output_a[:3000]
        pos_a_lbl, pos_b_lbl = label_b, label_a

    prompt = (
        PAIRWISE_COMPARISON_PROMPT.replace("{task}", task)
        .replace("{output_a}", pos_a_out)
        .replace("{output_b}", pos_b_out)
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        text = response.content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines[1:] if not line.strip().startswith("```")]
            text = "\n".join(lines)
        parsed = json.loads(text)
        ab_winner = parsed.get("winner", "A")
        winner = pos_a_lbl if ab_winner == "A" else pos_b_lbl
        return {
            "winner": winner,
            "confidence": parsed.get("confidence", "low"),
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception as exc:
        logger.warning("Pairwise comparison failed: %s", exc)
        return {"winner": label_a, "confidence": "low", "reasoning": f"error: {exc}"}


def _validate_on_holdout(
    llm: BaseChatModel,
    settings,
    prompt_store: PromptStore,
    memory_store,
    candidate_prompt: str,
    holdout_tasks: list[str],
) -> bool:
    """Pairwise gate run against a stable held-out task set.

    For each held-out task, runs BOTH the current champion and the
    candidate fresh (with the same harness state), then pairwise compares.
    This is the apples-to-apples gate: both outputs come from the same
    cycle and same harness, so the comparison isolates the prompt change.

    The held-out set is sourced from settings.tasks_file (split via
    ``_split_tasks``) and is stable across cycles, so trends are
    comparable and the optimizer can't overfit by validating on the
    exact tasks it generated the candidate against.
    """
    from src.evolution.orchestrator import _run_single_task

    wins_new = 0
    wins_old = 0
    errors = 0

    for task in holdout_tasks:
        # Run champion fresh (uses store's current prompt)
        old_result = _run_single_task(settings, prompt_store, memory_store, task)
        old_output = old_result.get("output", "")
        if not old_output or old_result.get("status") == "error":
            errors += 1
            wins_old += 1  # treat champion failure as a tie loss for safety
            continue

        # Run candidate fresh (with prompt override)
        new_result = _run_single_task(
            settings, prompt_store, memory_store, task, prompt_override=candidate_prompt
        )
        new_output = new_result.get("output", "")
        if not new_output or new_result.get("status") == "error":
            errors += 1
            wins_old += 1
            continue

        comparison = pairwise_compare(llm, task, old_output, new_output, "old", "new")
        logger.info(
            "Pairwise [holdout]: task='%s' winner=%s confidence=%s",
            task[:50],
            comparison["winner"],
            comparison["confidence"],
        )
        if comparison["winner"] == "new":
            wins_new += 1
        else:
            wins_old += 1

    logger.info(
        "Pairwise holdout validation: new=%d, old=%d (errors=%d, n=%d)",
        wins_new,
        wins_old,
        errors,
        len(holdout_tasks),
    )
    return wins_new > wins_old


def validate_candidate_prompt(
    llm: BaseChatModel,
    settings,
    prompt_store: PromptStore,
    memory_store,
    candidate_prompt: str,
    analyses: list[AnalysisResult],
    holdout_tasks: list[str] | None = None,
) -> bool:
    """Check that candidate prompt improves outputs via pairwise comparison.

    Two evaluation modes:

    1. **Held-out (preferred)** — when ``holdout_tasks`` is provided with
       at least 2 tasks, both champion and candidate are run fresh on the
       same held-out set and pairwise compared. This is apples-to-apples,
       stable across cycles, and not overfit to the optimizer's input.

    2. **Training-batch fallback** — when no holdout set is supplied,
       picks failing tasks from the current batch and compares the
       candidate's fresh output against the champion's *historical*
       output from the original run. This is the legacy behavior and is
       overfit-prone (the candidate was generated from these exact
       failures), so it logs a warning when this path runs.
    """
    from src.evolution.orchestrator import _run_single_task

    # --- Preferred path: held-out validation ---
    if holdout_tasks and len(holdout_tasks) >= 2:
        return _validate_on_holdout(
            llm, settings, prompt_store, memory_store, candidate_prompt, holdout_tasks
        )

    # --- Fallback: training-batch failures (legacy behavior) ---
    logger.warning(
        "Pairwise validation falling back to training-batch failures "
        "(no holdout set provided). This is overfit-prone — the candidate "
        "was generated from these exact tasks. Configure settings.tasks_file "
        "with a populated tasks JSON to enable held-out validation."
    )

    selected = _select_tasks_for_mini_scoring(analyses, max_tasks=2)
    if len(selected) < 2:
        logger.info("Not enough tasks for pairwise validation, accepting candidate")
        return True

    wins_new = 0
    wins_old = 0

    for analysis in selected:
        task = analysis["task"]
        old_output = analysis["output"]
        result = _run_single_task(
            settings, prompt_store, memory_store, task, prompt_override=candidate_prompt
        )
        new_output = result.get("output", "")
        if not new_output or result.get("status") == "error":
            wins_old += 1
            continue
        comparison = pairwise_compare(llm, task, old_output, new_output, "old", "new")
        logger.info(
            "Pairwise [batch-fallback]: task='%s' winner=%s confidence=%s",
            task[:50],
            comparison["winner"],
            comparison["confidence"],
        )
        if comparison["winner"] == "new":
            wins_new += 1
        else:
            wins_old += 1

    logger.info("Pairwise validation: new=%d, old=%d", wins_new, wins_old)
    return wins_new > wins_old


def optimize_prompt(
    llm: BaseChatModel,
    prompt_store: PromptStore,
    analyses: list[AnalysisResult],
    skills_dir: Path | None = None,
    settings=None,
    memory_store=None,
    trace_fetcher=None,
    holdout_tasks: list[str] | None = None,
) -> int:
    """Run the full prompt optimization pipeline.

    Args:
        llm: Language model for generating improved prompts
        prompt_store: Store for versioned prompts
        analyses: Analysis results from the current cycle
        skills_dir: Optional path to skills directory for skill-aware optimization
        trace_fetcher: Optional TraceFetcher for rich LangSmith traces
        holdout_tasks: Optional stable held-out task list. When provided
            (≥2 tasks), the pairwise gate runs both champion and candidate
            fresh on this set instead of comparing against current-batch
            failure outputs. Strongly preferred — see validate_candidate_prompt.

    Returns:
        New prompt version number
    """
    failed = [a for a in analyses if a["classification"] in ("failed", "partial")]
    if not failed:
        logger.info("No failures to optimize against, keeping current prompt")
        return prompt_store.get_latest_version_number()

    failure_info = analyze_failures(analyses, trace_fetcher=trace_fetcher)

    current_prompt = prompt_store.get_current_prompt()
    all_scores = [a["average_score"] for a in analyses]
    current_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # Build skills summary for skill-aware prompt optimization
    skills_summary = ""
    if skills_dir and Path(skills_dir).exists():
        skills = SkillManager(Path(skills_dir)).discover()
        if skills:
            skill_lines = [f"- {s['name']}: {s['description']}" for s in skills.values()]
            skills_summary = "\n".join(skill_lines)

    improved_prompt = generate_improved_prompt(
        llm, current_prompt, current_score, failure_info, skills_summary
    )

    # Pairwise validation: only adopt if candidate beats current
    if settings and memory_store:
        is_better = validate_candidate_prompt(
            llm,
            settings,
            prompt_store,
            memory_store,
            improved_prompt,
            analyses,
            holdout_tasks=holdout_tasks,
        )
        if not is_better:
            logger.info("Candidate prompt lost pairwise validation, keeping current")
            return prompt_store.get_latest_version_number()
    else:
        logger.info("Pairwise validation skipped (settings/memory_store not provided)")

    parent_version = prompt_store.get_latest_version_number()
    feedback_summary = json.dumps(failure_info["common_issues"][:5])

    # Inherit the parent's frozen promotion_score so the new champion has
    # a stable baseline for the ratchet. The new version just won pairwise
    # validation against the parent, so the parent's promotion_score is the
    # safest floor we can claim without re-running a held-out eval. If the
    # parent has no promotion_score (e.g. seeded baseline), we fall back to
    # the current batch average.
    parent = prompt_store.get_version(parent_version) if parent_version else None
    inherited = (
        parent.promotion_score
        if parent and parent.promotion_score is not None
        else current_score
    )

    new_version = prompt_store.add_version(
        prompt=improved_prompt,
        score=None,
        parent_version=parent_version,
        feedback_summary=feedback_summary,
        promotion_score=inherited,
    )

    logger.info(
        "Generated prompt v%d from %d failure analyses (parent: v%d)",
        new_version.version,
        len(failed),
        parent_version,
    )
    return new_version.version
