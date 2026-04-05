"""Factual spot-check via web search."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from evoagent.core.parsing import parse_llm_json
from src.agent.prompts import CLAIM_SELECTION_PROMPT, FACT_CHECK_PROMPT

logger = logging.getLogger(__name__)

VERDICT_SCORES = {"corroborated": 1.0, "inconclusive": 0.5, "contradicted": 0.0}


def _select_verifiable_claims(llm: BaseChatModel, claims: list[str]) -> list[str]:
    try:
        claims_text = "\n".join(f"- {c}" for c in claims)
        prompt = CLAIM_SELECTION_PROMPT.replace("{claims}", claims_text)
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = parse_llm_json(response.content)
        selected = parsed.get("selected", [])
        return selected if selected else claims[:2]
    except Exception:
        logger.warning("Failed to select verifiable claims", exc_info=True)
        return claims[:2]


def _check_single_claim(llm: BaseChatModel, search_tool: BaseTool, claim: str) -> str | None:
    try:
        search_results = search_tool._run(claim)
    except Exception:
        logger.warning("Search failed for claim: %s", claim, exc_info=True)
        return None

    if not search_results:
        return None

    search_text = str(search_results) if not isinstance(search_results, str) else search_results

    try:
        prompt = FACT_CHECK_PROMPT.replace("{claim}", claim).replace(
            "{search_results}", search_text[:3000]
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = parse_llm_json(response.content)
        verdict = parsed.get("verdict", "")
        return verdict if verdict in VERDICT_SCORES else "inconclusive"
    except Exception:
        logger.warning("LLM fact-check failed for claim: %s", claim, exc_info=True)
        return None


def _compute_spot_check_score(verdicts: list[str]) -> float | None:
    if not verdicts:
        return None
    return sum(VERDICT_SCORES.get(v, 0.5) for v in verdicts) / len(verdicts)


def spot_check_claims(llm: BaseChatModel, search_tool: BaseTool, claims: list[str]) -> float | None:
    """Spot-check claims by searching the web."""
    if not claims:
        return None
    selected = _select_verifiable_claims(llm, claims)
    verdicts: list[str] = []
    for claim in selected:
        verdict = _check_single_claim(llm, search_tool, claim)
        if verdict is not None:
            verdicts.append(verdict)
    return _compute_spot_check_score(verdicts)
