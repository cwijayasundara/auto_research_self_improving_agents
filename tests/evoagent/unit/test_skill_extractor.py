"""Tests for unified skill extraction."""

from unittest.mock import MagicMock

from evoagent.skills.extractor import extract_skills_from_batch, is_duplicate_skill
from evoagent.skills.manager import SkillManager


def test_is_duplicate_exact_match():
    existing = {"retry-search": {}, "handle-timeout": {}}
    assert is_duplicate_skill("retry-search", existing) is True


def test_is_duplicate_word_overlap():
    existing = {"retry-search-queries": {}}
    assert is_duplicate_skill("retry-search", existing) is True


def test_not_duplicate():
    existing = {"retry-search": {}}
    assert is_duplicate_skill("handle-timeout", existing) is False


def test_extract_skips_low_score(tmp_path):
    mgr = SkillManager(tmp_path)
    analyses = [
        {
            "run_id": "r1",
            "task": "test",
            "classification": "failed",
            "average_score": 0.3,
            "grader_results": [],
            "output": "bad output",
            "tool_calls": [],
        }
    ]
    created = extract_skills_from_batch(
        llm=MagicMock(), analyses=analyses, skill_store=mgr, mode="success"
    )
    assert created == []
