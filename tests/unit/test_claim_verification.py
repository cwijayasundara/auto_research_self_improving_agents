"""Tests for claim-level verification grader."""

import json
from unittest.mock import MagicMock

import pytest

from src.evolution.graders.claim_verification import (
    _compute_consistency_score,
    _extract_claims,
    _parse_json_response,
    _verify_claims,
    grade_claims,
)


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def test_plain_json(self):
        text = '{"claims": ["claim 1", "claim 2"]}'
        result = _parse_json_response(text)
        assert result == {"claims": ["claim 1", "claim 2"]}

    def test_markdown_code_block(self):
        text = '```json\n{"claims": ["claim 1"]}\n```'
        result = _parse_json_response(text)
        assert result == {"claims": ["claim 1"]}

    def test_invalid_json_returns_empty(self):
        result = _parse_json_response("not json at all")
        assert result == {}

    def test_empty_string(self):
        result = _parse_json_response("")
        assert result == {}


# ---------------------------------------------------------------------------
# _compute_consistency_score
# ---------------------------------------------------------------------------

class TestComputeConsistencyScore:
    def test_all_supported(self):
        verdicts = [
            {"claim": "a", "verdict": "supported", "reasoning": "ok"},
            {"claim": "b", "verdict": "supported", "reasoning": "ok"},
            {"claim": "c", "verdict": "supported", "reasoning": "ok"},
        ]
        assert _compute_consistency_score(verdicts) == 1.0

    def test_mixed_verdicts(self):
        verdicts = [
            {"claim": "a", "verdict": "supported", "reasoning": "ok"},
            {"claim": "b", "verdict": "unsupported", "reasoning": "no source"},
            {"claim": "c", "verdict": "contradicted", "reasoning": "conflict"},
        ]
        assert _compute_consistency_score(verdicts) == pytest.approx(1 / 3)

    def test_empty_returns_half(self):
        assert _compute_consistency_score([]) == 0.5


# ---------------------------------------------------------------------------
# _extract_claims
# ---------------------------------------------------------------------------

class TestExtractClaims:
    def test_happy_path(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='{"claims": ["The sky is blue", "Water is wet"]}'
        )
        claims = _extract_claims(llm, "The sky is blue. Water is wet.")
        assert claims == ["The sky is blue", "Water is wet"]
        # Verify LLM was called
        llm.invoke.assert_called_once()

    def test_parse_failure_returns_empty(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="not valid json")
        claims = _extract_claims(llm, "some output")
        assert claims == []

    def test_exception_returns_empty(self):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("LLM broke")
        claims = _extract_claims(llm, "some output")
        assert claims == []


# ---------------------------------------------------------------------------
# _verify_claims
# ---------------------------------------------------------------------------

class TestVerifyClaims:
    def test_happy_path(self):
        llm = MagicMock()
        verdicts = [
            {"claim": "The sky is blue", "verdict": "supported", "reasoning": "cited"},
        ]
        llm.invoke.return_value = MagicMock(
            content=json.dumps({"verdicts": verdicts})
        )
        result = _verify_claims(llm, ["The sky is blue"], "The sky is blue per NASA.")
        assert len(result) == 1
        assert result[0]["verdict"] == "supported"

    def test_parse_failure_returns_empty(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="garbage")
        result = _verify_claims(llm, ["claim"], "output")
        assert result == []

    def test_exception_returns_empty(self):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("LLM broke")
        result = _verify_claims(llm, ["claim"], "output")
        assert result == []


# ---------------------------------------------------------------------------
# grade_claims (full pipeline)
# ---------------------------------------------------------------------------

class TestGradeClaims:
    def test_full_pipeline(self):
        llm = MagicMock()
        # First call: extract claims
        llm.invoke.side_effect = [
            MagicMock(
                content='{"claims": ["claim A", "claim B"]}'
            ),
            # Second call: verify claims
            MagicMock(
                content=json.dumps({
                    "verdicts": [
                        {"claim": "claim A", "verdict": "supported", "reasoning": "ok"},
                        {"claim": "claim B", "verdict": "supported", "reasoning": "ok"},
                    ]
                })
            ),
        ]
        result = grade_claims(llm, "task", "output text")
        assert result["name"] == "claim_verification"
        assert result["score"] == 1.0
        assert result["passed"] is True

    def test_no_claims_extracted(self):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="bad response")
        result = grade_claims(llm, "task", "output")
        assert result["score"] == 0.5
        assert result["passed"] is False

    def test_no_verdicts_returned(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(content='{"claims": ["claim A"]}'),
            MagicMock(content="bad response"),
        ]
        result = grade_claims(llm, "task", "output")
        assert result["score"] == 0.5
        assert result["passed"] is False

    def test_spot_check_blending(self):
        llm = MagicMock()
        # consistency = 1.0 (all supported), spot_check = 0.5
        # final = 0.6 * 1.0 + 0.4 * 0.5 = 0.8
        llm.invoke.side_effect = [
            MagicMock(
                content='{"claims": ["claim A"]}'
            ),
            MagicMock(
                content=json.dumps({
                    "verdicts": [
                        {"claim": "claim A", "verdict": "supported", "reasoning": "ok"},
                    ]
                })
            ),
        ]
        result = grade_claims(llm, "task", "output", spot_check_score=0.5)
        assert result["score"] == pytest.approx(0.8)
        assert result["passed"] is True

    def test_low_score_fails(self):
        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(
                content='{"claims": ["claim A", "claim B", "claim C"]}'
            ),
            MagicMock(
                content=json.dumps({
                    "verdicts": [
                        {"claim": "claim A", "verdict": "contradicted", "reasoning": "x"},
                        {"claim": "claim B", "verdict": "unsupported", "reasoning": "x"},
                        {"claim": "claim C", "verdict": "unsupported", "reasoning": "x"},
                    ]
                })
            ),
        ]
        result = grade_claims(llm, "task", "output")
        assert result["score"] == pytest.approx(0.0)
        assert result["passed"] is False
