"""Application settings loaded from environment variables.

Uses pydantic-settings BaseSettings to validate and load configuration.
All secrets come from environment variables or .env files - never hardcoded.
"""

import logging
import os
from pathlib import Path

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API keys — set the ones needed for your chosen provider
    openai_api_key: str = ""
    tavily_api_key: str = ""
    ollama_api_key: str = ""
    langchain_api_key: str = ""
    langsmith_api_key: str = ""

    # LangSmith tracing
    langchain_tracing_v2: bool = True
    langsmith_project: str = "auto-research-self-improving"

    # Model configuration
    model: str = "gpt-4o"
    model_provider: str = "openai"
    model_base_url: str = ""

    # Inner loop evolution settings
    max_evolution_cycles: int = 5
    batch_size: int = 3

    # Outer loop (autoresearch) settings
    outer_loop_enabled: bool = True
    max_inner_cycles: int = 5
    min_improvement_threshold: float = 0.05

    # Directory paths (relative to project root)
    skills_dir: str = "skills"
    prompts_dir: str = "prompts"
    memory_dir: str = "memory"
    evolution_state_dir: str = "evolution_state"

    # Memory compression
    memory_token_budget: int = 1000
    compression_similarity_threshold: float = 0.7

    # Sub-agents
    use_subagents: bool = True

    # Logging
    log_level: str = "INFO"

    @property
    def resolved_api_key(self) -> str:
        """Return whichever LangSmith/LangChain API key is set."""
        return self.langchain_api_key or self.langsmith_api_key

    @property
    def skills_path(self) -> Path:
        return PROJECT_ROOT / self.skills_dir

    @property
    def prompts_path(self) -> Path:
        return PROJECT_ROOT / self.prompts_dir

    @property
    def memory_path(self) -> Path:
        return PROJECT_ROOT / self.memory_dir

    @property
    def evolution_state_path(self) -> Path:
        return PROJECT_ROOT / self.evolution_state_dir


def load_settings() -> Settings | None:
    """Load settings, returning None if validation fails."""
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        logger.error(
            "Configuration error: %s. "
            "Ensure required environment variables are set. See .env.example.",
            exc,
        )
        return None


def export_langsmith_env(settings: Settings) -> None:
    """Export LangSmith env vars so langchain auto-tracing picks them up."""
    api_key = settings.resolved_api_key
    os.environ.setdefault("LANGCHAIN_TRACING_V2", str(settings.langchain_tracing_v2).lower())
    if api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
        os.environ.setdefault("LANGSMITH_API_KEY", api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    if settings.openai_api_key:
        os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    if settings.tavily_api_key:
        os.environ.setdefault("TAVILY_API_KEY", settings.tavily_api_key)
    if settings.ollama_api_key:
        os.environ.setdefault("OLLAMA_API_KEY", settings.ollama_api_key)
    logger.info("LangSmith tracing enabled: project='%s'", settings.langsmith_project)


def configure_logging(settings: Settings) -> None:
    """Configure logging based on settings.log_level."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
