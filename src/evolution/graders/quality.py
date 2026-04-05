"""LLM-as-judge grader: output quality."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evoagent.core.parsing import parse_llm_json
from evoagent.core.types import GraderResult
from src.agent.prompts import QUALITY_PROMPT

logger = logging.getLogger(__name__)


def grade_quality(llm: BaseChatModel, task: str, output: str) -> GraderResult:
    """Grade the quality of the agent's output."""
    prompt = QUALITY_PROMPT.format(task=task, output=output[:4000])
    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = parse_llm_json(response.content)

    score = float(parsed.get("overall_score", 0.5))
    return GraderResult(
        name="quality",
        score=round(score, 3),
        passed=score >= 0.75,
        reasoning=parsed.get("reasoning", ""),
    )
