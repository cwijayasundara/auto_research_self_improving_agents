"""Metaprompt-based prompt optimization."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evoagent.core.protocols import PromptStore

logger = logging.getLogger(__name__)

MAX_AUTONOMY_RETRIES = 2

_AUTONOMY_VIOLATIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ask\s+(the\s+)?user", re.I), "asks the user"),
    (re.compile(r"would you like", re.I), "asks 'would you like'"),
    (re.compile(r"do you (?:want|prefer|need)", re.I), "asks user preference"),
    (re.compile(r"let me know", re.I), "requests user feedback"),
]

DEFAULT_METAPROMPT = (
    "You are a prompt engineering expert.\n\n"
    "## Current Prompt\n{current_prompt}\n\n"
    "## Current Score\n{current_score}\n\n"
    "## Failure Analysis\n{failure_analysis}\n\n"
    "## Common Issues\n{common_issues}\n\n"
    "Generate an improved system prompt that addresses the failures "
    "while preserving what works. Output ONLY the new prompt text."
)


def validate_prompt_autonomy(prompt: str) -> list[str]:
    """Check for patterns that violate autonomous operation."""
    return [desc for pattern, desc in _AUTONOMY_VIOLATIONS if pattern.search(prompt)]


def optimize_prompt(
    llm: BaseChatModel,
    prompt_store: PromptStore,
    analyses: list[dict[str, Any]],
    metaprompt_template: str | None = None,
) -> int:
    """Generate an improved prompt. Returns new version number."""
    version, current_prompt = prompt_store.get_current()

    failed = [a for a in analyses if a.get("classification") in ("failed", "partial")]
    if not failed:
        return version

    issues: list[str] = []
    for a in failed:
        for g in a.get("grader_results", []):
            if isinstance(g, dict) and not g.get("passed", True):
                issues.append(f"[{g.get('name', '?')}] {g.get('reasoning', '')[:150]}")

    template = metaprompt_template or DEFAULT_METAPROMPT
    scores = [a.get("average_score", 0) for a in analyses]
    avg_score = sum(scores) / len(scores) if scores else 0

    metaprompt = template.format(
        current_prompt=current_prompt[:3000],
        current_score=f"{avg_score:.3f}",
        failure_analysis=f"{len(failed)} failed/partial out of {len(analyses)}",
        common_issues="\n".join(f"- {i}" for i in issues[:10]),
    )

    for attempt in range(1 + MAX_AUTONOMY_RETRIES):
        response = llm.invoke([HumanMessage(content=metaprompt)])
        new_prompt = response.content.strip()

        violations = validate_prompt_autonomy(new_prompt)
        if not violations:
            return prompt_store.save(new_prompt, score=None, parent=version)

        logger.warning("Autonomy violations (attempt %d): %s", attempt + 1, violations)
        metaprompt += f"\n\nThe previous attempt had violations: {violations}. Fix them."

    logger.error("Failed to generate autonomous prompt after %d attempts", MAX_AUTONOMY_RETRIES + 1)
    return version
