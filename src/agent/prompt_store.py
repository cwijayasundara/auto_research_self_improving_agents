"""Versioned prompt persistence.

Stores prompt versions as JSON files in the prompts/ directory.
Supports version history, scoring, and rollback.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class PromptVersion:
    """A single versioned prompt with metadata."""

    def __init__(
        self,
        version: int,
        prompt: str,
        score: float | None = None,
        timestamp: str | None = None,
        parent_version: int | None = None,
        feedback_summary: str = "",
    ) -> None:
        self.version = version
        self.prompt = prompt
        self.score = score
        self.timestamp = timestamp or datetime.now(UTC).isoformat()
        self.parent_version = parent_version
        self.feedback_summary = feedback_summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "prompt": self.prompt,
            "score": self.score,
            "timestamp": self.timestamp,
            "parent_version": self.parent_version,
            "feedback_summary": self.feedback_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptVersion":
        return cls(
            version=data["version"],
            prompt=data["prompt"],
            score=data.get("score"),
            timestamp=data.get("timestamp"),
            parent_version=data.get("parent_version"),
            feedback_summary=data.get("feedback_summary", ""),
        )


class PromptStore:
    """File-based versioned prompt store."""

    def __init__(self, prompts_dir: Path) -> None:
        self.prompts_dir = prompts_dir
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def _version_path(self, version: int) -> Path:
        return self.prompts_dir / f"v{version:04d}.json"

    def get_all_versions(self) -> list[PromptVersion]:
        """Load all prompt versions sorted by version number."""
        versions = []
        for path in sorted(self.prompts_dir.glob("v*.json")):
            with open(path) as f:
                data = json.load(f)
            versions.append(PromptVersion.from_dict(data))
        return versions

    def get_version(self, version: int) -> PromptVersion | None:
        """Get a specific prompt version."""
        path = self._version_path(version)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return PromptVersion.from_dict(data)

    def get_current_prompt(self) -> str:
        """Return the best-scoring prompt, or the latest if no scores exist."""
        versions = self.get_all_versions()
        if not versions:
            return ""
        scored = [v for v in versions if v.score is not None]
        if scored:
            best = max(scored, key=lambda v: v.score)  # type: ignore[arg-type]
            return best.prompt
        return versions[-1].prompt

    def get_latest_version_number(self) -> int:
        """Return the latest version number, or 0 if no versions exist."""
        versions = self.get_all_versions()
        if not versions:
            return 0
        return versions[-1].version

    def add_version(
        self,
        prompt: str,
        score: float | None = None,
        parent_version: int | None = None,
        feedback_summary: str = "",
    ) -> PromptVersion:
        """Add a new prompt version."""
        version = self.get_latest_version_number() + 1
        pv = PromptVersion(
            version=version,
            prompt=prompt,
            score=score,
            parent_version=parent_version,
            feedback_summary=feedback_summary,
        )
        path = self._version_path(version)
        with open(path, "w") as f:
            json.dump(pv.to_dict(), f, indent=2)
        logger.info("Saved prompt version %d (score=%s)", version, score)
        return pv

    def update_score(self, version: int, score: float) -> None:
        """Update the score for an existing prompt version.

        Uses incremental averaging: new_avg = old_avg + (score - old_avg) / n.
        This way every run contributes to the prompt's score without needing
        to store all individual scores.
        """
        pv = self.get_version(version)
        if pv is None:
            logger.warning("Cannot update score: version %d not found", version)
            return

        path = self._version_path(version)
        # Load raw dict to access score_count (not in PromptVersion dataclass)
        with open(path) as f:
            data = json.load(f)

        n = data.get("score_count", 1 if pv.score is not None else 0)
        if n == 0 or pv.score is None:
            pv.score = score
            n = 1
        else:
            n += 1
            pv.score = pv.score + (score - pv.score) / n

        data["score"] = round(pv.score, 4)
        data["score_count"] = n
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Updated score for version %d to %.4f (n=%d)", version, pv.score, n)

    def append_feedback(self, version: int, feedback: str) -> None:
        """Append a feedback entry to an existing prompt version."""
        path = self._version_path(version)
        if not path.exists():
            logger.warning("Cannot append feedback: version %d not found", version)
            return
        with open(path) as f:
            data = json.load(f)
        existing = data.get("feedback_summary", "")
        # Keep feedback compact — cap at ~500 chars
        if len(existing) > 500:
            return
        separator = "; " if existing else ""
        data["feedback_summary"] = existing + separator + feedback
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Appended feedback to version %d", version)
