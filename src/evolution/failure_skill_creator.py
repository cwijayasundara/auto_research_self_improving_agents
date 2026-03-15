"""Failure-driven skill creator.

Analyzes failed trajectories and creates defensive skills that teach
the agent how to avoid similar failures in future runs. This is the
key innovation combining:
- Anthropic's skill-creator pattern (structured SKILL.md with assertive triggers)
- Self-evolving agents' failure analysis (grader-driven pattern detection)
- Autoresearch's learn-from-mistakes philosophy (keep what works, learn from what doesn't)

Unlike the standard skill_extractor (which learns from successes), this module
learns from failures — creating "antibody" skills that prevent recurring issues.
"""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agent.prompts import FAILURE_SKILL_PROMPT
from src.evolution.state import AnalysisResult
from src.skills.manager import create_skill, discover_skills, validate_skill

logger = logging.getLogger(__name__)

# Minimum number of failed trajectories to trigger failure skill creation
MIN_FAILURES_FOR_SKILL = 2

# Maximum failure skills to create per cycle (prevent skill explosion)
MAX_FAILURE_SKILLS_PER_CYCLE = 3


def _parse_failure_skill_response(text: str) -> dict[str, Any]:
    """Parse JSON from failure skill creation LLM response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse failure skill response")
        return {}


def _group_failures_by_pattern(
    analyses: list[AnalysisResult],
) -> dict[str, list[AnalysisResult]]:
    """Group failed trajectories by their primary failure grader.

    Returns a dict mapping failure pattern keys to the analyses that share
    that pattern, enabling targeted skill creation per failure type.
    """
    groups: dict[str, list[AnalysisResult]] = {}

    for analysis in analyses:
        if analysis["classification"] not in ("failed", "partial"):
            continue

        # Find the worst-performing grader as the primary failure signal
        failed_graders = [
            g for g in analysis["grader_results"] if not g["passed"]
        ]
        if not failed_graders:
            continue

        # Group by the grader with lowest score
        worst = min(failed_graders, key=lambda g: g["score"])
        pattern_key = worst["name"]
        groups.setdefault(pattern_key, []).append(analysis)

    return groups


def _build_failure_context(analyses: list[AnalysisResult]) -> str:
    """Build a context string summarizing failed trajectories."""
    parts: list[str] = []
    for i, analysis in enumerate(analyses[:5], 1):  # Cap at 5 to fit context
        grader_summary = "\n".join(
            f"    - {g['name']}: {g['score']:.2f} ({'PASS' if g['passed'] else 'FAIL'}) "
            f"— {g['reasoning'][:100]}"
            for g in analysis["grader_results"]
        )
        parts.append(
            f"### Failure {i}\n"
            f"  Task: {analysis['task'][:200]}\n"
            f"  Classification: {analysis['classification']}\n"
            f"  Score: {analysis['average_score']:.3f}\n"
            f"  Grader Results:\n{grader_summary}\n"
            f"  Output Preview: {analysis['output'][:300]}..."
        )
    return "\n\n".join(parts)


def _extract_failure_patterns(analyses: list[AnalysisResult]) -> str:
    """Extract and deduplicate failure patterns across analyses."""
    patterns: list[str] = []
    for analysis in analyses:
        for grader in analysis["grader_results"]:
            if not grader["passed"]:
                pattern = (
                    f"[{grader['name']}] score={grader['score']:.2f}: "
                    f"{grader['reasoning'][:150]}"
                )
                if pattern not in patterns:
                    patterns.append(pattern)
    return "\n".join(f"- {p}" for p in patterns[:10])


def create_failure_skill(
    llm: BaseChatModel,
    failure_group: list[AnalysisResult],
    pattern_name: str,
    skills_dir: Path,
) -> Path | None:
    """Create a defensive skill from a group of related failures.

    Args:
        llm: Language model for generating skill content
        failure_group: List of analyses sharing a failure pattern
        pattern_name: Name of the failure pattern (e.g., 'task_completion')
        skills_dir: Directory to store skills

    Returns:
        Path to created SKILL.md, or None if creation failed
    """
    existing = discover_skills(skills_dir)

    # Check if we already have a skill for this failure pattern
    defensive_name = f"handle-{pattern_name}-failures"
    if defensive_name in existing:
        logger.info("Defensive skill '%s' already exists, skipping", defensive_name)
        return None

    failed_context = _build_failure_context(failure_group)
    failure_patterns = _extract_failure_patterns(failure_group)

    prompt = FAILURE_SKILL_PROMPT.format(
        failed_trajectories=failed_context,
        failure_patterns=failure_patterns,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = _parse_failure_skill_response(response.content)

    if not parsed.get("name") or not parsed.get("content"):
        logger.warning("Failure skill creation returned incomplete data for pattern '%s'", pattern_name)
        return None

    skill_name = parsed["name"]
    description = parsed.get(
        "description",
        f"Defensive skill preventing {pattern_name} failures. "
        f"Use when the agent encounters similar task patterns.",
    )

    skill_path = create_skill(
        skills_dir=skills_dir,
        name=skill_name,
        description=description,
        content=parsed["content"],
    )

    is_valid, msg = validate_skill(skill_path)
    if not is_valid:
        logger.warning("Created failure skill failed validation: %s", msg)
        skill_path.unlink()
        return None

    logger.info(
        "Created defensive skill '%s' from %d failures (pattern: %s)",
        skill_name, len(failure_group), pattern_name,
    )
    return skill_path


def create_failure_skills_from_batch(
    llm: BaseChatModel,
    analyses: list[AnalysisResult],
    skills_dir: Path,
) -> list[Path]:
    """Create defensive skills from all failure patterns in a batch.

    Groups failures by pattern, then creates one skill per pattern group
    (up to MAX_FAILURE_SKILLS_PER_CYCLE to prevent skill explosion).

    Returns:
        List of paths to newly created failure skill files.
    """
    failure_groups = _group_failures_by_pattern(analyses)

    eligible_groups = {
        pattern: group
        for pattern, group in failure_groups.items()
        if len(group) >= MIN_FAILURES_FOR_SKILL
    }

    if not eligible_groups:
        logger.info("No failure patterns meet threshold (%d+ failures required)", MIN_FAILURES_FOR_SKILL)
        return []

    logger.info(
        "Creating defensive skills for %d failure patterns: %s",
        len(eligible_groups),
        ", ".join(eligible_groups.keys()),
    )

    created: list[Path] = []
    for pattern_name, group in list(eligible_groups.items())[:MAX_FAILURE_SKILLS_PER_CYCLE]:
        path = create_failure_skill(llm, group, pattern_name, skills_dir)
        if path:
            created.append(path)

    return created
