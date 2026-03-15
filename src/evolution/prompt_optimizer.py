"""Prompt optimizer using metaprompt approach.

Enhanced with skill-awareness: the optimizer can reference learned skills
when generating improved prompts, creating tighter integration between
the skill learning and prompt optimization loops.
"""

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agent.prompt_store import PromptStore
from src.agent.prompts import METAPROMPT_TEMPLATE
from src.evolution.state import AnalysisResult
from src.skills.manager import discover_skills

logger = logging.getLogger(__name__)


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
                    f"[{grader['name']}] {grader['reasoning']} "
                    f"(task: {analysis['task'][:80]})"
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
    """
    prompt = METAPROMPT_TEMPLATE.format(
        current_prompt=current_prompt,
        current_score=f"{current_score:.3f}",
        failure_analysis=failure_info["failure_analysis"],
        common_issues="\n".join(f"- {i}" for i in failure_info["common_issues"]),
        available_skills=skills_summary or "No skills learned yet.",
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    improved = response.content.strip()

    if "{memory_context}" not in improved:
        improved += "\n{memory_context}"

    return improved


def optimize_prompt(
    llm: BaseChatModel,
    prompt_store: PromptStore,
    analyses: list[AnalysisResult],
    skills_dir: "Path | None" = None,
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
    from pathlib import Path

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
            skill_lines = [
                f"- {s['name']}: {s['description']}" for s in skills.values()
            ]
            skills_summary = "\n".join(skill_lines)

    improved_prompt = generate_improved_prompt(
        llm, current_prompt, current_score, failure_info, skills_summary
    )

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
        new_version.version, len(failed), parent_version,
    )
    return new_version.version
