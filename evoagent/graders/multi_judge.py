"""Multi-judge grading implementing Grader protocol."""

from __future__ import annotations

import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evoagent.core.parsing import parse_llm_json
from evoagent.core.protocols import Grader
from evoagent.core.types import GraderResult

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_PROMPT = (
    "Rate the following agent output on a scale of 0.0 to 1.0.\n\n"
    "Task: {task}\n\nOutput:\n{output}\n\n"
    'Respond as JSON: {{"score": <float>, "reasoning": "<string>"}}'
)


class MultiJudgeGrader(Grader):
    """Run multiple LLM judges in parallel and aggregate via median."""

    def __init__(
        self,
        llm: BaseChatModel,
        name: str = "quality",
        judge_prompts: list[str] | None = None,
        fallback_prompt: str | None = None,
        pass_threshold: float = 0.75,
    ) -> None:
        self.name = name
        self._llm = llm
        self._judge_prompts = judge_prompts or [DEFAULT_JUDGE_PROMPT] * 3
        self._fallback_prompt = fallback_prompt or DEFAULT_JUDGE_PROMPT
        self._pass_threshold = pass_threshold

    def grade(self, task: str, output: str, **kwargs: Any) -> GraderResult:
        score, reasoning = self._run_multi_judge(task, output)
        return GraderResult(
            name=self.name,
            score=score,
            passed=score >= self._pass_threshold,
            reasoning=reasoning,
        )

    def _run_single_judge(
        self, prompt_template: str, task: str, output: str
    ) -> tuple[float | None, str]:
        try:
            prompt = prompt_template.replace("{task}", task).replace("{output}", output[:4000])
            response = self._llm.invoke([HumanMessage(content=prompt)])
            parsed = parse_llm_json(response.content)
            if "score" not in parsed:
                return 0.5, f"parse_error: no score key in {response.content[:200]}"
            return float(parsed["score"]), parsed.get("reasoning", "")
        except Exception as exc:
            return None, str(exc)

    def _run_multi_judge(self, task: str, output: str) -> tuple[float, str]:
        scores: list[float] = []
        reasonings: list[str] = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._run_single_judge, p, task, output): i
                for i, p in enumerate(self._judge_prompts)
            }
            for future in as_completed(futures):
                score, reasoning = future.result()
                if score is not None:
                    scores.append(score)
                    reasonings.append(reasoning)

        if len(scores) >= 2:
            median_score = statistics.median(scores)
            combined = " | ".join(reasonings)
            spread = max(scores) - min(scores)
            if spread > 0.3:
                combined = f"[low_agreement spread={spread:.2f}] {combined}"
            return median_score, combined

        logger.warning(
            "Only %d/%d judges succeeded, falling back",
            len(scores),
            len(self._judge_prompts),
        )
        score, reasoning = self._run_single_judge(self._fallback_prompt, task, output)
        if score is not None:
            return score, reasoning
        return 0.5, "grader_error: all judges and fallback failed"
