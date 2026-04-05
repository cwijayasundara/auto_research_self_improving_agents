"""Unified skill extraction from successful and failed trajectories."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from evoagent.core.parsing import parse_llm_json
from evoagent.skills.manager import SkillManager

logger = logging.getLogger(__name__)

SUCCESS_THRESHOLD = 0.70
FAILURE_THRESHOLD = 0.50
MAX_SKILLS_PER_CYCLE = 3

DEFAULT_SUCCESS_PROMPT = (
    "Analyze this successful agent trajectory and extract a reusable skill.\n\n"
    "Task: {task}\nOutput (preview): {output}\nScore: {score}\n\n"
    'Respond as JSON: {{"name": "...", "description": "...", "content": "..."}}'
)

DEFAULT_FAILURE_PROMPT = (
    "Analyze these failed trajectories and create a defensive skill.\n\n"
    "Failures:\n{failures}\n\nFailure patterns:\n{patterns}\n\n"
    'Respond as JSON: {{"name": "...", "description": "...", "content": "..."}}'
)


def is_duplicate_skill(name: str, existing_skills: dict[str, Any]) -> bool:
    normalized = name.lower().replace(" ", "-").replace("_", "-")
    for skill_id in existing_skills:
        if skill_id == normalized:
            return True
        existing_words = set(skill_id.split("-"))
        new_words = set(normalized.split("-"))
        if len(existing_words & new_words) >= 2:
            return True
    return False


def extract_skills_from_batch(
    llm: Any,
    analyses: list[dict[str, Any]],
    skill_store: SkillManager,
    mode: str = "success",
    prompt_template: str | None = None,
) -> list[Path]:
    from langchain_core.messages import HumanMessage

    if mode == "success":
        eligible = [a for a in analyses if a.get("average_score", 0) >= SUCCESS_THRESHOLD]
        template = prompt_template or DEFAULT_SUCCESS_PROMPT
    else:
        eligible = [a for a in analyses if a.get("average_score", 0) < FAILURE_THRESHOLD]
        template = prompt_template or DEFAULT_FAILURE_PROMPT

    if not eligible:
        return []

    existing = skill_store.discover()
    created: list[Path] = []

    for analysis in eligible[:MAX_SKILLS_PER_CYCLE]:
        try:
            if mode == "success":
                prompt = template.format(
                    task=analysis.get("task", ""),
                    output=analysis.get("output", "")[:3000],
                    score=analysis.get("average_score", 0),
                )
            else:
                prompt = template.format(
                    failures=analysis.get("output", "")[:2000],
                    patterns=str(analysis.get("grader_results", []))[:1000],
                )

            response = llm.invoke([HumanMessage(content=prompt)])
            parsed = parse_llm_json(response.content)

            if not parsed.get("name") or not parsed.get("content"):
                continue

            if is_duplicate_skill(parsed["name"], existing):
                continue

            path = skill_store.create(
                name=parsed["name"],
                description=parsed.get("description", f"Extracted from {mode} run"),
                content=parsed["content"],
            )
            is_valid, msg = skill_store.validate(path)
            if not is_valid:
                path.unlink()
                continue

            created.append(path)
            existing[parsed["name"].lower().replace(" ", "-")] = {}

        except Exception as exc:
            logger.error("Skill extraction failed: %s", exc)

    return created
