"""LLM-as-judge grader: output quality."""

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agent.prompts import QUALITY_PROMPT
from src.evolution.state import GraderResult

logger = logging.getLogger(__name__)


def _parse_quality_response(text: str) -> dict[str, Any]:
    """Parse JSON from quality grader LLM response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse quality grader response")
        return {"overall_score": 0.5, "reasoning": cleaned}


def grade_quality(
    llm: BaseChatModel,
    task: str,
    output: str,
) -> GraderResult:
    """Grade the quality of the agent's output."""
    prompt = QUALITY_PROMPT.format(task=task, output=output[:4000])
    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = _parse_quality_response(response.content)

    score = float(parsed.get("overall_score", 0.5))
    return GraderResult(
        name="quality",
        score=round(score, 3),
        passed=score >= 0.75,
        reasoning=parsed.get("reasoning", ""),
    )
