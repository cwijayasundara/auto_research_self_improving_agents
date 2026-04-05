"""Base configuration for EvoAgent.

Provides sensible defaults. Users can subclass or override via env vars.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class EvoAgentConfig(BaseSettings):
    """Configuration for the evoagent framework."""

    model_config = {"extra": "ignore"}

    # Base directory for all storage
    base_dir: str = "."

    # Evolution loop
    max_cycles: int = 5
    batch_size: int = 3
    min_improvement_threshold: float = 0.05
    plateau_cycles: int = 2
    task_timeout_seconds: int = 300

    # Memory
    memory_token_budget: int = 4000
    compression_similarity_threshold: float = 0.7

    # Directory names (relative to base_dir)
    memory_dir: str = "memory"
    skills_dir: str = "skills"
    prompts_dir: str = "prompts"
    traces_dir: str = "traces"
    evolution_state_dir: str = "evolution_state"

    @property
    def _base(self) -> Path:
        return Path(self.base_dir)

    @property
    def memory_path(self) -> Path:
        return self._base / self.memory_dir

    @property
    def skills_path(self) -> Path:
        return self._base / self.skills_dir

    @property
    def prompts_path(self) -> Path:
        return self._base / self.prompts_dir

    @property
    def traces_path(self) -> Path:
        return self._base / self.traces_dir

    @property
    def evolution_state_path(self) -> Path:
        return self._base / self.evolution_state_dir
