"""Tests for grading prompt rubric anchors."""

import pytest

from src.agent.prompts import (
    Q_DEPTH_PROMPT,
    Q_RELEVANCE_PROMPT,
    Q_STRUCTURE_PROMPT,
    QUALITY_PROMPT,
    TASK_COMPLETION_PROMPT,
    TC_ACCURACY_PROMPT,
    TC_COMPLETENESS_PROMPT,
    TC_EVIDENCE_PROMPT,
)


class TestTaskCompletionPromptAnchors:
    """TASK_COMPLETION_PROMPT must contain rubric anchors at 0.9, 0.5, 0.2."""

    def test_contains_scoring_guide_section(self):
        assert "## Scoring Guide" in TASK_COMPLETION_PROMPT

    def test_contains_0_9_anchor(self):
        assert "0.9" in TASK_COMPLETION_PROMPT

    def test_contains_0_5_anchor(self):
        assert "0.5" in TASK_COMPLETION_PROMPT

    def test_contains_0_2_anchor(self):
        assert "0.2" in TASK_COMPLETION_PROMPT

    def test_0_9_anchor_mentions_cited_sources(self):
        # The 0.9 anchor should reference cited sources
        assert "cited sources" in TASK_COMPLETION_PROMPT

    def test_0_2_anchor_mentions_off_topic(self):
        assert "off-topic" in TASK_COMPLETION_PROMPT.lower()

    def test_uses_judging_language(self):
        assert "judging" in TASK_COMPLETION_PROMPT.lower()

    def test_preserves_existing_criteria(self):
        assert "core question" in TASK_COMPLETION_PROMPT
        assert "well-structured" in TASK_COMPLETION_PROMPT
        assert "cited sources" in TASK_COMPLETION_PROMPT

    def test_preserves_json_response_format(self):
        assert "Respond as JSON" in TASK_COMPLETION_PROMPT
        assert "score:" in TASK_COMPLETION_PROMPT
        assert "passed:" in TASK_COMPLETION_PROMPT
        assert "reasoning:" in TASK_COMPLETION_PROMPT


class TestQualityPromptAnchors:
    """QUALITY_PROMPT must contain rubric anchors at 0.9, 0.5, 0.2."""

    def test_contains_scoring_guide_section(self):
        assert "## Scoring Guide" in QUALITY_PROMPT

    def test_contains_0_9_anchor(self):
        assert "0.9" in QUALITY_PROMPT

    def test_contains_0_5_anchor(self):
        assert "0.5" in QUALITY_PROMPT

    def test_contains_0_2_anchor(self):
        assert "0.2" in QUALITY_PROMPT

    def test_0_9_anchor_mentions_citations(self):
        assert "citations" in QUALITY_PROMPT.lower()

    def test_0_9_anchor_mentions_logical_structure(self):
        assert "logical structure" in QUALITY_PROMPT.lower()

    def test_0_2_anchor_mentions_disorganized(self):
        assert "disorganized" in QUALITY_PROMPT.lower()

    def test_uses_judging_language(self):
        assert "judging" in QUALITY_PROMPT.lower()

    def test_preserves_existing_dimensions(self):
        assert "Accuracy" in QUALITY_PROMPT
        assert "Depth" in QUALITY_PROMPT
        assert "Clarity" in QUALITY_PROMPT
        assert "Relevance" in QUALITY_PROMPT

    def test_preserves_json_response_format(self):
        assert "Respond as JSON" in QUALITY_PROMPT
        assert "overall_score:" in QUALITY_PROMPT
        assert "reasoning:" in QUALITY_PROMPT


# ---------------------------------------------------------------------------
# Perspective judge prompts
# ---------------------------------------------------------------------------

PERSPECTIVE_PROMPTS = [
    ("TC_COMPLETENESS_PROMPT", TC_COMPLETENESS_PROMPT),
    ("TC_EVIDENCE_PROMPT", TC_EVIDENCE_PROMPT),
    ("TC_ACCURACY_PROMPT", TC_ACCURACY_PROMPT),
    ("Q_STRUCTURE_PROMPT", Q_STRUCTURE_PROMPT),
    ("Q_DEPTH_PROMPT", Q_DEPTH_PROMPT),
    ("Q_RELEVANCE_PROMPT", Q_RELEVANCE_PROMPT),
]


class TestPerspectiveJudgePromptsPlaceholders:
    """All 6 perspective prompts must have {task} and {output} placeholders."""

    @pytest.mark.parametrize("name,prompt", PERSPECTIVE_PROMPTS)
    def test_has_task_placeholder(self, name, prompt):
        assert "{task}" in prompt, f"{name} missing {{task}} placeholder"

    @pytest.mark.parametrize("name,prompt", PERSPECTIVE_PROMPTS)
    def test_has_output_placeholder(self, name, prompt):
        assert "{output}" in prompt, f"{name} missing {{output}} placeholder"


class TestPerspectiveJudgePromptsAnchors:
    """All 6 perspective prompts must have score anchors at 0.9, 0.5, 0.2."""

    @pytest.mark.parametrize("name,prompt", PERSPECTIVE_PROMPTS)
    def test_has_0_9_anchor(self, name, prompt):
        assert "0.9" in prompt, f"{name} missing 0.9 anchor"

    @pytest.mark.parametrize("name,prompt", PERSPECTIVE_PROMPTS)
    def test_has_0_5_anchor(self, name, prompt):
        assert "0.5" in prompt, f"{name} missing 0.5 anchor"

    @pytest.mark.parametrize("name,prompt", PERSPECTIVE_PROMPTS)
    def test_has_0_2_anchor(self, name, prompt):
        assert "0.2" in prompt, f"{name} missing 0.2 anchor"


class TestPerspectiveJudgePromptsJSON:
    """All 6 perspective prompts must request a JSON response."""

    @pytest.mark.parametrize("name,prompt", PERSPECTIVE_PROMPTS)
    def test_requests_json_response(self, name, prompt):
        assert "JSON" in prompt, f"{name} missing JSON response request"

    @pytest.mark.parametrize("name,prompt", PERSPECTIVE_PROMPTS)
    def test_json_has_score_key(self, name, prompt):
        assert '"score"' in prompt, f"{name} missing score key in JSON"

    @pytest.mark.parametrize("name,prompt", PERSPECTIVE_PROMPTS)
    def test_json_has_reasoning_key(self, name, prompt):
        assert '"reasoning"' in prompt, f"{name} missing reasoning key in JSON"
