"""Tests for factual spot-check via web search."""

import json
from unittest.mock import MagicMock

import pytest

from src.evolution.graders.fact_checker import (
    VERDICT_SCORES,
    _check_single_claim,
    _compute_spot_check_score,
    _parse_json_response,
    _select_verifiable_claims,
    spot_check_claims,
)


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_plain_json(self):
        text = '{"selected": ["claim 1", "claim 2"]}'
        result = _parse_json_response(text)
        assert result == {"selected": ["claim 1", "claim 2"]}

    def test_markdown_code_block(self):
        text = '```json\n{"selected": ["claim 1"]}\n```'
        result = _parse_json_response(text)
        assert result == {"selected": ["claim 1"]}

    def test_invalid_json_returns_empty(self):
        result = _parse_json_response("not json at all")
        assert result == {}

    def test_empty_string(self):
        result = _parse_json_response("")
        assert result == {}


# ---------------------------------------------------------------------------
# _compute_spot_check_score
# ---------------------------------------------------------------------------

class TestComputeSpotCheckScore:
    def test_all_corroborated(self):
        verdicts = ["corroborated", "corroborated", "corroborated"]
        assert _compute_spot_check_score(verdicts) == 1.0

    def test_all_contradicted(self):
        verdicts = ["contradicted", "contradicted"]
        assert _compute_spot_check_score(verdicts) == 0.0

    def test_mixed_verdicts(self):
        verdicts = ["corroborated", "contradicted"]
        assert _compute_spot_check_score(verdicts) == pytest.approx(0.5)

    def test_empty_returns_none(self):
        assert _compute_spot_check_score([]) is None

    def test_all_inconclusive(self):
        verdicts = ["inconclusive", "inconclusive"]
        assert _compute_spot_check_score(verdicts) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _select_verifiable_claims
# ---------------------------------------------------------------------------

class TestSelectVerifiableClaims:
    def test_happy_path(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='{"selected": ["GDP grew by 3.2% in 2024", "Paris has 2.1M residents"]}'
        )
        claims = [
            "GDP grew by 3.2% in 2024",
            "Paris has 2.1M residents",
            "The economy is doing well",
        ]
        result = _select_verifiable_claims(llm, claims)
        assert result == ["GDP grew by 3.2% in 2024", "Paris has 2.1M residents"]
        llm.invoke.assert_called_once()

    def test_parse_failure_falls_back_to_first_two(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="not valid json")
        claims = ["claim A", "claim B", "claim C"]
        result = _select_verifiable_claims(llm, claims)
        assert result == ["claim A", "claim B"]

    def test_exception_falls_back_to_first_two(self):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("LLM broke")
        claims = ["claim A", "claim B", "claim C"]
        result = _select_verifiable_claims(llm, claims)
        assert result == ["claim A", "claim B"]


# ---------------------------------------------------------------------------
# _check_single_claim
# ---------------------------------------------------------------------------

class TestCheckSingleClaim:
    def test_corroborated_path(self):
        llm = MagicMock()
        search_tool = MagicMock()
        search_tool._run.return_value = "Search result: GDP grew 3.2% in 2024 per BEA."

        llm.invoke.return_value = MagicMock(
            content=json.dumps({
                "verdict": "corroborated",
                "reasoning": "BEA confirms the figure",
            })
        )
        result = _check_single_claim(llm, search_tool, "GDP grew by 3.2% in 2024")
        assert result == "corroborated"
        search_tool._run.assert_called_once_with("GDP grew by 3.2% in 2024")

    def test_search_failure_returns_none(self):
        llm = MagicMock()
        search_tool = MagicMock()
        search_tool._run.side_effect = Exception("Search API down")

        result = _check_single_claim(llm, search_tool, "some claim")
        assert result is None
        llm.invoke.assert_not_called()

    def test_empty_search_results_returns_none(self):
        llm = MagicMock()
        search_tool = MagicMock()
        search_tool._run.return_value = ""

        result = _check_single_claim(llm, search_tool, "some claim")
        assert result is None

    def test_unknown_verdict_returns_inconclusive(self):
        llm = MagicMock()
        search_tool = MagicMock()
        search_tool._run.return_value = "Some search results"

        llm.invoke.return_value = MagicMock(
            content=json.dumps({
                "verdict": "maybe",
                "reasoning": "unclear",
            })
        )
        result = _check_single_claim(llm, search_tool, "some claim")
        assert result == "inconclusive"

    def test_llm_failure_returns_none(self):
        llm = MagicMock()
        search_tool = MagicMock()
        search_tool._run.return_value = "Some search results"
        llm.invoke.side_effect = Exception("LLM broke")

        result = _check_single_claim(llm, search_tool, "some claim")
        assert result is None


# ---------------------------------------------------------------------------
# spot_check_claims (full pipeline)
# ---------------------------------------------------------------------------

class TestSpotCheckClaims:
    def test_no_claims_returns_none(self):
        llm = MagicMock()
        search_tool = MagicMock()
        result = spot_check_claims(llm, search_tool, [])
        assert result is None

    def test_all_searches_fail_returns_none(self):
        llm = MagicMock()
        search_tool = MagicMock()
        search_tool._run.side_effect = Exception("Search API down")

        # LLM call for claim selection
        llm.invoke.return_value = MagicMock(
            content='{"selected": ["claim A", "claim B"]}'
        )

        result = spot_check_claims(llm, search_tool, ["claim A", "claim B", "claim C"])
        assert result is None

    def test_happy_path_all_corroborated(self):
        llm = MagicMock()
        search_tool = MagicMock()
        search_tool._run.return_value = "Confirming evidence"

        # First call: select claims; subsequent calls: fact-check each
        llm.invoke.side_effect = [
            MagicMock(content='{"selected": ["claim A", "claim B"]}'),
            MagicMock(content=json.dumps({"verdict": "corroborated", "reasoning": "ok"})),
            MagicMock(content=json.dumps({"verdict": "corroborated", "reasoning": "ok"})),
        ]

        result = spot_check_claims(llm, search_tool, ["claim A", "claim B", "claim C"])
        assert result == pytest.approx(1.0)

    def test_mixed_verdicts(self):
        llm = MagicMock()
        search_tool = MagicMock()
        search_tool._run.return_value = "Some evidence"

        llm.invoke.side_effect = [
            MagicMock(content='{"selected": ["claim A", "claim B"]}'),
            MagicMock(content=json.dumps({"verdict": "corroborated", "reasoning": "ok"})),
            MagicMock(content=json.dumps({"verdict": "contradicted", "reasoning": "wrong"})),
        ]

        result = spot_check_claims(llm, search_tool, ["claim A", "claim B"])
        assert result == pytest.approx(0.5)
