"""Claim-level verification grader."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evoagent.core.parsing import parse_llm_json
from evoagent.core.types import GraderResult
from src.agent.prompts import CLAIM_EXTRACTION_PROMPT, CLAIM_VERIFICATION_PROMPT

logger = logging.getLogger(__name__)


def _extract_claims(llm: BaseChatModel, output: str) -> list[str]:
    try:
        prompt = CLAIM_EXTRACTION_PROMPT.replace("{output}", output[:4000])
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = parse_llm_json(response.content)
        return parsed.get("claims", [])
    except Exception:
        logger.warning("Failed to extract claims", exc_info=True)
        return []


def _verify_claims(llm: BaseChatModel, claims: list[str], output: str) -> list[dict]:
    try:
        claims_text = "\n".join(f"- {c}" for c in claims)
        prompt = CLAIM_VERIFICATION_PROMPT.replace("{claims}", claims_text).replace(
            "{output}", output[:4000]
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = parse_llm_json(response.content)
        return parsed.get("verdicts", [])
    except Exception:
        logger.warning("Failed to verify claims", exc_info=True)
        return []


def _compute_consistency_score(verdicts: list[dict]) -> float:
    if not verdicts:
        return 0.5
    supported = sum(1 for v in verdicts if v.get("verdict") == "supported")
    return supported / len(verdicts)


def grade_claims(
    llm: BaseChatModel,
    task: str,
    output: str,
    spot_check_score: float | None = None,
) -> GraderResult:
    """Grade agent output by extracting and verifying factual claims."""
    claims = _extract_claims(llm, output)
    if not claims:
        return GraderResult(
            name="claim_verification", score=0.5, passed=False,
            reasoning="No claims could be extracted from the output.",
        )

    verdicts = _verify_claims(llm, claims, output)
    if not verdicts:
        return GraderResult(
            name="claim_verification", score=0.5, passed=False,
            reasoning="Claim verification failed to produce verdicts.",
        )

    consistency = _compute_consistency_score(verdicts)
    final = 0.6 * consistency + 0.4 * spot_check_score if spot_check_score is not None else consistency

    supported_count = sum(1 for v in verdicts if v.get("verdict") == "supported")
    return GraderResult(
        name="claim_verification",
        score=final,
        passed=final >= 0.6,
        reasoning=(
            f"Consistency: {consistency:.2f} ({supported_count}/{len(verdicts)} supported). "
            + (f"Spot-check: {spot_check_score:.2f}. " if spot_check_score is not None else "")
            + f"Final: {final:.2f}."
        ),
    )
