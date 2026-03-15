"""LLM-as-judge grader: task completion."""

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agent.prompts import TASK_COMPLETION_PROMPT
from src.evolution.state import GraderResult

logger = logging.getLogger(__name__)


def _parse_grader_response(text: str) -> dict[str, Any]:
    """Parse JSON from grader LLM response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse task completion grader response")
        return {"score": 0.5, "passed": False, "reasoning": cleaned}


def grade_task_completion(
    llm: BaseChatModel,
    task: str,
    output: str,
) -> GraderResult:
    """Grade whether the agent completed the task."""
    prompt = TASK_COMPLETION_PROMPT.format(task=task, output=output[:4000])
    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = _parse_grader_response(response.content)

    score = float(parsed.get("score", 0.5))
    return GraderResult(
        name="task_completion",
        score=score,
        passed=parsed.get("passed", score >= 0.75),
        reasoning=parsed.get("reasoning", ""),
    )
