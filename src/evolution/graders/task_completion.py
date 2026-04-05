"""LLM-as-judge grader: task completion."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evoagent.core.parsing import parse_llm_json
from evoagent.core.types import GraderResult
from src.agent.prompts import TASK_COMPLETION_PROMPT

logger = logging.getLogger(__name__)


def grade_task_completion(llm: BaseChatModel, task: str, output: str) -> GraderResult:
    """Grade whether the agent completed the task."""
    prompt = TASK_COMPLETION_PROMPT.format(task=task, output=output[:4000])
    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = parse_llm_json(response.content)

    score = float(parsed.get("score", 0.5))
    return GraderResult(
        name="task_completion",
        score=score,
        passed=parsed.get("passed", score >= 0.75),
        reasoning=parsed.get("reasoning", ""),
    )
