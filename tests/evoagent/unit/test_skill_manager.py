"""Tests for SkillManager."""

from evoagent.skills.manager import SkillManager


def test_discover_empty(tmp_path):
    mgr = SkillManager(tmp_path)
    assert mgr.discover() == {}


def test_create_and_discover(tmp_path):
    mgr = SkillManager(tmp_path)
    path = mgr.create("retry-search", "Retry failed searches", "# Instructions\nRetry with backoff")
    assert path.exists()
    skills = mgr.discover()
    assert "retry-search" in skills
    assert skills["retry-search"]["name"] == "retry-search"


def test_load_skill(tmp_path):
    mgr = SkillManager(tmp_path)
    mgr.create("my-skill", "Test skill", "# Steps\n1. Do thing")
    content = mgr.load("my-skill")
    assert content is not None
    assert "Do thing" in content


def test_load_missing_skill(tmp_path):
    mgr = SkillManager(tmp_path)
    assert mgr.load("nonexistent") is None


def test_validate_skill(tmp_path):
    mgr = SkillManager(tmp_path)
    path = mgr.create("valid", "A valid skill", "# Content\nSome body text")
    is_valid, msg = mgr.validate(path)
    assert is_valid
