"""Tests for grading prompt rubric anchors."""

from src.agent.prompts import TASK_COMPLETION_PROMPT, QUALITY_PROMPT


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
