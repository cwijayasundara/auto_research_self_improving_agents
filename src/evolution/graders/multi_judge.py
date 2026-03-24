"""Multi-judge grading with parallel perspective judges.

Runs 3 perspective-specific judges in parallel via ThreadPoolExecutor,
aggregates scores via median, and falls back to a single-judge prompt
if too many judges fail.
"""

import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agent.prompts import (
    QUALITY_PROMPT,
    Q_DEPTH_PROMPT,
    Q_RELEVANCE_PROMPT,
    Q_STRUCTURE_PROMPT,
    TASK_COMPLETION_PROMPT,
    TC_ACCURACY_PROMPT,
    TC_COMPLETENESS_PROMPT,
    TC_EVIDENCE_PROMPT,
)
from src.evolution.state import GraderResult

logger = logging.getLogger(__name__)


def _parse_judge_response(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return {}


def _run_single_judge(
    llm: BaseChatModel,
    prompt_template: str,
    task: str,
    output: str,
) -> tuple[float | None, str]:
    """Run a single judge: format prompt, call LLM, parse response.

    Returns:
        (score, reasoning) on success.
        (0.5, "parse_error:...") if JSON lacks "score" key.
        (None, error_msg) on exception.
    """
    try:
        prompt = prompt_template.replace("{task}", task).replace(
            "{output}", output[:4000]
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_judge_response(response.content)

        if "score" not in parsed:
            return 0.5, f"parse_error: no score key in response: {response.content[:200]}"

        return float(parsed["score"]), parsed.get("reasoning", "")
    except Exception as exc:
        return None, str(exc)


def _aggregate_judge_scores(
    scores: list[float],
    reasonings: list[str],
) -> tuple[float, str]:
    """Aggregate multiple judge scores via median.

    Returns:
        (score, combined_reasoning). Flags "low_agreement" if max-min > 0.3.
        Empty list -> (0.5, "grader_error: no judge scores available").
    """
    if not scores:
        return 0.5, "grader_error: no judge scores available"

    if len(scores) == 1:
        return scores[0], reasonings[0]

    median_score = statistics.median(scores)
    spread = max(scores) - min(scores)
    combined = " | ".join(reasonings)

    if spread > 0.3:
        combined = f"[low_agreement spread={spread:.2f}] {combined}"

    return median_score, combined


def _run_multi_judge(
    llm: BaseChatModel,
    judge_prompts: list[str],
    fallback_prompt: str,
    task: str,
    output: str,
) -> tuple[float, str]:
    """Run multiple judges in parallel, aggregate, fall back if needed.

    Runs all judges via ThreadPoolExecutor(max_workers=3).
    If 2+ succeed -> aggregate via median.
    If <2 succeed -> retry with single fallback prompt.
    If fallback also fails -> return (0.5, "grader_error: ...").
    """
    scores: list[float] = []
    reasonings: list[str] = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_run_single_judge, llm, prompt, task, output): i
            for i, prompt in enumerate(judge_prompts)
        }
        for future in as_completed(futures):
            score, reasoning = future.result()
            if score is not None:
                scores.append(score)
                reasonings.append(reasoning)

    if len(scores) >= 2:
        return _aggregate_judge_scores(scores, reasonings)

    # Fallback: not enough judges succeeded
    logger.warning(
        "Only %d/%d judges succeeded, falling back to single prompt",
        len(scores),
        len(judge_prompts),
    )
    fallback_score, fallback_reasoning = _run_single_judge(
        llm, fallback_prompt, task, output
    )
    if fallback_score is not None:
        return fallback_score, fallback_reasoning

    return 0.5, "grader_error: all judges and fallback failed"


def multi_judge_task_completion(
    llm: BaseChatModel,
    task: str,
    output: str,
) -> GraderResult:
    """Grade task completion using 3 perspective judges (completeness, evidence, accuracy)."""
    score, reasoning = _run_multi_judge(
        llm,
        [TC_COMPLETENESS_PROMPT, TC_EVIDENCE_PROMPT, TC_ACCURACY_PROMPT],
        TASK_COMPLETION_PROMPT,
        task,
        output,
    )
    return GraderResult(
        name="task_completion",
        score=score,
        passed=score >= 0.75,
        reasoning=reasoning,
    )


def multi_judge_quality(
    llm: BaseChatModel,
    task: str,
    output: str,
) -> GraderResult:
    """Grade output quality using 3 perspective judges (structure, depth, relevance)."""
    score, reasoning = _run_multi_judge(
        llm,
        [Q_STRUCTURE_PROMPT, Q_DEPTH_PROMPT, Q_RELEVANCE_PROMPT],
        QUALITY_PROMPT,
        task,
        output,
    )
    return GraderResult(
        name="quality",
        score=score,
        passed=score >= 0.75,
        reasoning=reasoning,
    )
