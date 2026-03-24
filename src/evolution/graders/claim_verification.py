"""Claim-level verification grader.

Extracts factual claims from agent output, verifies them for internal
consistency and source alignment, and produces a GraderResult.
"""

import json
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.agent.prompts import CLAIM_EXTRACTION_PROMPT, CLAIM_VERIFICATION_PROMPT
from src.evolution.state import GraderResult

logger = logging.getLogger(__name__)


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks.

    Returns empty dict on failure.
    """
    cleaned = text.strip()
    if not cleaned:
        return {}
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines[1:] if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Failed to parse claim verification JSON response")
        return {}


def _extract_claims(llm: BaseChatModel, output: str) -> list[str]:
    """Extract factual claims from agent output via LLM.

    Returns list of claim strings, or [] on failure.
    """
    try:
        prompt = CLAIM_EXTRACTION_PROMPT.replace("{output}", output[:4000])
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_json_response(response.content)
        return parsed.get("claims", [])
    except Exception:
        logger.warning("Failed to extract claims", exc_info=True)
        return []


def _verify_claims(llm: BaseChatModel, claims: list[str], output: str) -> list[dict]:
    """Verify each claim against the source output via LLM.

    Returns list of verdict dicts, or [] on failure.
    """
    try:
        claims_text = "\n".join(f"- {c}" for c in claims)
        prompt = CLAIM_VERIFICATION_PROMPT.replace("{claims}", claims_text).replace(
            "{output}", output[:4000]
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_json_response(response.content)
        return parsed.get("verdicts", [])
    except Exception:
        logger.warning("Failed to verify claims", exc_info=True)
        return []


def _compute_consistency_score(verdicts: list[dict]) -> float:
    """Compute fraction of supported verdicts. Returns 0.5 if empty."""
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
    """Grade agent output by extracting and verifying factual claims.

    Args:
        llm: Language model for claim extraction and verification.
        task: The research task description.
        output: The agent's output text.
        spot_check_score: Optional external spot-check score to blend in.

    Returns:
        GraderResult with name="claim_verification".
    """
    # Extract claims
    claims = _extract_claims(llm, output)
    if not claims:
        return GraderResult(
            name="claim_verification",
            score=0.5,
            passed=False,
            reasoning="No claims could be extracted from the output.",
        )

    # Verify claims
    verdicts = _verify_claims(llm, claims, output)
    if not verdicts:
        return GraderResult(
            name="claim_verification",
            score=0.5,
            passed=False,
            reasoning="Claim verification failed to produce verdicts.",
        )

    # Compute score
    consistency = _compute_consistency_score(verdicts)

    if spot_check_score is not None:
        final = 0.6 * consistency + 0.4 * spot_check_score
    else:
        final = consistency

    return GraderResult(
        name="claim_verification",
        score=final,
        passed=final >= 0.6,
        reasoning=(
            f"Consistency: {consistency:.2f} "
            f"({sum(1 for v in verdicts if v.get('verdict') == 'supported')}"
            f"/{len(verdicts)} supported). "
            + (f"Spot-check: {spot_check_score:.2f}. " if spot_check_score is not None else "")
            + f"Final: {final:.2f}."
        ),
    )
