"""Factual spot-check via web search.

Selects the most verifiable claims from agent output, searches the web for
each, and compares search results against the claims. Returns a spot-check
score (float or None) that can be blended into the claim_verification grader.
"""

import json
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from src.agent.prompts import CLAIM_SELECTION_PROMPT, FACT_CHECK_PROMPT

logger = logging.getLogger(__name__)

VERDICT_SCORES = {"corroborated": 1.0, "inconclusive": 0.5, "contradicted": 0.0}


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
        logger.warning("Failed to parse fact-check JSON response")
        return {}


def _select_verifiable_claims(llm: BaseChatModel, claims: list[str]) -> list[str]:
    """Select the most objectively verifiable claims via LLM.

    On failure, falls back to the first 2 claims.
    """
    try:
        claims_text = "\n".join(f"- {c}" for c in claims)
        prompt = CLAIM_SELECTION_PROMPT.replace("{claims}", claims_text)
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_json_response(response.content)
        selected = parsed.get("selected", [])
        if selected:
            return selected
        return claims[:2]
    except Exception:
        logger.warning("Failed to select verifiable claims", exc_info=True)
        return claims[:2]


def _check_single_claim(llm: BaseChatModel, search_tool: BaseTool, claim: str) -> str | None:
    """Check a single claim against web search results.

    Returns verdict string ('corroborated', 'contradicted', 'inconclusive'),
    or None if search or LLM fails.
    """
    # Search for evidence
    try:
        search_results = search_tool._run(claim)
    except Exception:
        logger.warning("Search failed for claim: %s", claim, exc_info=True)
        return None

    if not search_results:
        return None

    # Ask LLM to compare claim against search results
    try:
        prompt = FACT_CHECK_PROMPT.replace("{claim}", claim).replace(
            "{search_results}", search_results[:3000]
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_json_response(response.content)
        verdict = parsed.get("verdict", "")
        if verdict in VERDICT_SCORES:
            return verdict
        return "inconclusive"
    except Exception:
        logger.warning("LLM fact-check failed for claim: %s", claim, exc_info=True)
        return None


def _compute_spot_check_score(verdicts: list[str]) -> float | None:
    """Compute mean score from verdict strings.

    Returns None if the list is empty.
    """
    if not verdicts:
        return None
    return sum(VERDICT_SCORES.get(v, 0.5) for v in verdicts) / len(verdicts)


def spot_check_claims(llm: BaseChatModel, search_tool: BaseTool, claims: list[str]) -> float | None:
    """Spot-check claims by searching the web for each.

    Args:
        llm: Language model for claim selection and verdict.
        search_tool: A LangChain BaseTool for web search.
        claims: List of claim strings extracted from agent output.

    Returns:
        Float score between 0.0 and 1.0, or None if no verdicts could be produced.
    """
    if not claims:
        return None

    selected = _select_verifiable_claims(llm, claims)

    verdicts: list[str] = []
    for claim in selected:
        verdict = _check_single_claim(llm, search_tool, claim)
        if verdict is not None:
            verdicts.append(verdict)

    return _compute_spot_check_score(verdicts)
