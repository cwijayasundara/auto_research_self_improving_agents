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
from src.skills.manager import discover_skills

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


def analyze_failures(analyses: list[AnalysisResult]) -> dict[str, Any]:
    """Aggregate failure patterns from failed/partial trajectories."""
    failed = [a for a in analyses if a["classification"] in ("failed", "partial")]
    if not failed:
        return {"failure_analysis": "No failures to analyze.", "common_issues": []}

    issues: list[str] = []
    for analysis in failed:
        for grader in analysis["grader_results"]:
            if not grader["passed"]:
                issues.append(
                    f"[{grader['name']}] {grader['reasoning']} (task: {analysis['task'][:80]})"
                )

    unique_issues = list(dict.fromkeys(issues))

    failure_summary = (
        f"Analyzed {len(failed)} failed/partial trajectories. "
        f"Found {len(unique_issues)} distinct issues."
    )

    return {
        "failure_analysis": failure_summary,
        "common_issues": unique_issues[:10],
    }


MAX_AUTONOMY_RETRIES = 2


def generate_improved_prompt(
    llm: BaseChatModel,
    current_prompt: str,
    current_score: float,
    failure_info: dict[str, Any],
    skills_summary: str = "",
) -> str:
    """Generate an improved prompt using the metaprompt approach.

    Enhanced: includes a summary of available skills so the prompt
    can reference them for better integration.

    Includes autonomy validation — if the generated prompt contains patterns
    that would cause the agent to ask the user for input, it retries up to
    MAX_AUTONOMY_RETRIES times, then falls back to the current prompt.
    """
    meta = METAPROMPT_TEMPLATE.format(
        current_prompt=current_prompt,
        current_score=f"{current_score:.3f}",
        failure_analysis=failure_info["failure_analysis"],
        common_issues="\n".join(f"- {i}" for i in failure_info["common_issues"]),
        available_skills=skills_summary or "No skills learned yet.",
    )

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

    # All retries failed — keep the current prompt to avoid drift
    logger.error(
        "Prompt optimizer failed autonomy validation after %d attempts. "
        "Keeping current prompt to prevent drift.",
        1 + MAX_AUTONOMY_RETRIES,
    )
    return current_prompt


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


def validate_candidate_prompt(
    llm: BaseChatModel,
    settings,
    prompt_store: PromptStore,
    memory_store,
    candidate_prompt: str,
    analyses: list[AnalysisResult],
) -> bool:
    """Check that candidate prompt improves outputs via pairwise comparison."""
    from src.evolution.orchestrator import _run_single_task

    selected = _select_tasks_for_mini_scoring(analyses, max_tasks=2)
    if len(selected) < 2:
        logger.info("Not enough tasks for pairwise validation, accepting candidate")
        return True

    # Temporarily add candidate prompt as latest version
    prompt_store.add_version(candidate_prompt, score=None)

    wins_new = 0
    wins_old = 0

    for analysis in selected:
        task = analysis["task"]
        old_output = analysis["output"]
        result = _run_single_task(settings, prompt_store, memory_store, task)
        new_output = result.get("output", "")
        if not new_output or result.get("status") == "error":
            wins_old += 1
            continue
        comparison = pairwise_compare(llm, task, old_output, new_output, "old", "new")
        logger.info(
            "Pairwise: task='%s' winner=%s confidence=%s",
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
) -> int:
    """Run the full prompt optimization pipeline.

    Args:
        llm: Language model for generating improved prompts
        prompt_store: Store for versioned prompts
        analyses: Analysis results from the current cycle
        skills_dir: Optional path to skills directory for skill-aware optimization

    Returns:
        New prompt version number
    """
    failed = [a for a in analyses if a["classification"] in ("failed", "partial")]
    if not failed:
        logger.info("No failures to optimize against, keeping current prompt")
        return prompt_store.get_latest_version_number()

    failure_info = analyze_failures(analyses)

    current_prompt = prompt_store.get_current_prompt()
    all_scores = [a["average_score"] for a in analyses]
    current_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # Build skills summary for skill-aware prompt optimization
    skills_summary = ""
    if skills_dir and Path(skills_dir).exists():
        skills = discover_skills(Path(skills_dir))
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
        )
        if not is_better:
            logger.info("Candidate prompt lost pairwise validation, keeping current")
            return prompt_store.get_latest_version_number()
    else:
        logger.info("Pairwise validation skipped (settings/memory_store not provided)")

    parent_version = prompt_store.get_latest_version_number()
    feedback_summary = json.dumps(failure_info["common_issues"][:5])

    new_version = prompt_store.add_version(
        prompt=improved_prompt,
        score=None,
        parent_version=parent_version,
        feedback_summary=feedback_summary,
    )

    logger.info(
        "Generated prompt v%d from %d failure analyses (parent: v%d)",
        new_version.version,
        len(failed),
        parent_version,
    )
    return new_version.version
