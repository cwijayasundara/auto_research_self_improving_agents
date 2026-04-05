"""Tests for EvoAgentConfig."""

from pathlib import Path

from evoagent.core.config import EvoAgentConfig


def test_default_config():
    config = EvoAgentConfig()
    assert config.max_cycles == 5
    assert config.min_improvement_threshold == 0.05
    assert config.memory_token_budget == 4000


def test_config_override():
    config = EvoAgentConfig(max_cycles=20, memory_token_budget=8000)
    assert config.max_cycles == 20
    assert config.memory_token_budget == 8000


def test_config_paths(tmp_path):
    config = EvoAgentConfig(base_dir=str(tmp_path))
    assert config.memory_path == tmp_path / "memory"
    assert config.skills_path == tmp_path / "skills"
    assert config.prompts_path == tmp_path / "prompts"
    assert config.traces_path == tmp_path / "traces"
