"""Skill extractor for high-scoring trajectories.

Extracts reusable skills from successful agent runs and stores them
as SKILL.md files following Anthropic's skill-creator conventions.
"""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agent.prompts import SKILL_EXTRACTION_PROMPT
from src.evolution.state import AnalysisResult
from src.skills.manager import create_skill, discover_skills, validate_skill

logger = logging.getLogger(__name__)

SKILL_EXTRACTION_THRESHOLD = 0.70


def _parse_skill_response(text: str) -> dict[str, Any]:
    """Parse JSON from skill extraction LLM response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse skill extraction response")
        return {}


def _is_duplicate_skill(name: str, existing_skills: dict[str, Any]) -> bool:
    """Check if a skill with a similar name already exists."""
    normalized = name.lower().replace(" ", "-").replace("_", "-")
    for skill_id in existing_skills:
        if skill_id == normalized:
            return True
        existing_words = set(skill_id.split("-"))
        new_words = set(normalized.split("-"))
        common = existing_words & new_words
        if len(common) >= 2:
            return True
    return False


def extract_skill(
    llm: BaseChatModel,
    analysis: AnalysisResult,
    skills_dir: Path,
) -> Path | None:
    """Extract a skill from a successful trajectory."""
    score = analysis["average_score"]
    if score < SKILL_EXTRACTION_THRESHOLD:
        return None

    existing = discover_skills(skills_dir)
    tool_calls_str = json.dumps(analysis["tool_calls"][:15], indent=2)

    prompt = SKILL_EXTRACTION_PROMPT.format(
        task=analysis["task"],
        tool_calls=tool_calls_str,
        output=analysis["output"][:3000],
        score=analysis["average_score"],
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = _parse_skill_response(response.content)

    if not parsed.get("name") or not parsed.get("content"):
        logger.warning("Skill extraction returned incomplete data")
        return None

    skill_name = parsed["name"]
    if _is_duplicate_skill(skill_name, existing):
        logger.info("Skill '%s' is a duplicate, skipping", skill_name)
        return None

    skill_path = create_skill(
        skills_dir=skills_dir,
        name=skill_name,
        description=parsed.get("description", "Extracted from successful run"),
        content=parsed["content"],
    )

    is_valid, msg = validate_skill(skill_path)
    if not is_valid:
        logger.warning("Created skill failed validation: %s", msg)
        skill_path.unlink()
        return None

    logger.info("Extracted skill '%s' from run %s", skill_name, analysis["run_id"])
    return skill_path


def extract_skills_from_batch(
    llm: BaseChatModel,
    analyses: list[AnalysisResult],
    skills_dir: Path,
) -> list[Path]:
    """Extract skills from all successful trajectories in a batch."""
    eligible = [a for a in analyses if a["average_score"] >= SKILL_EXTRACTION_THRESHOLD]
    logger.info(
        "Extracting skills from %d eligible trajectories (score >= %.2f) out of %d total",
        len(eligible),
        SKILL_EXTRACTION_THRESHOLD,
        len(analyses),
    )

    created: list[Path] = []
    for analysis in eligible:
        path = extract_skill(llm, analysis, skills_dir)
        if path:
            created.append(path)
    return created
