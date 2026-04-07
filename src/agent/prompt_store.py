"""Versioned prompt persistence.

Stores prompt versions as JSON files in the prompts/ directory.

Implements a Karpathy-style "ratchet" where promotion is monotonic:

- ``promotion_score`` is FROZEN at promotion time. It is set when a new
  version is added (typically inherited from the parent's promotion score,
  since the new version just won pairwise validation against it). Once set,
  it never changes.
- ``score`` is the running observation: an incremental average of post-
  promotion run scores. It can drift up or down with new runs and is used
  by the optimizer to decide *when* to try optimizing, but it is NOT used
  to decide which version is "current".
- The active version is always the LATEST version (highest version number),
  period. Promotions only move forward. The system never silently rolls
  back to an older version because new runs happened to drag the current
  version's running average below an older version's.

This closes three leaks in the previous implementation:
1. Score dilution dragging a champion below an older version's frozen score.
2. ``get_current_prompt`` selecting on a moving target (the running average).
3. Bad runs (or bugs producing phantom signal) silently changing which
   version is active without any optimizer involvement.
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
        promotion_score: float | None = None,
    ) -> None:
        self.version = version
        self.prompt = prompt
        # Running observation; mutated by update_score
        self.score = score
        # Frozen promotion baseline; set once at add_version time
        self.promotion_score = promotion_score
        self.timestamp = timestamp or datetime.now(UTC).isoformat()
        self.parent_version = parent_version
        self.feedback_summary = feedback_summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "prompt": self.prompt,
            "score": self.score,
            "promotion_score": self.promotion_score,
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
            promotion_score=data.get("promotion_score"),
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
        """Return the LATEST prompt version (the current champion).

        This is the ratchet: promotions only move forward. The active
        version is whichever was promoted most recently, period. We do NOT
        select by max-score across versions, because doing so would let
        running-average drift on the current champion silently roll the
        active version back to an older one.
        """
        versions = self.get_all_versions()
        if not versions:
            return ""
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
        promotion_score: float | None = None,
    ) -> PromptVersion:
        """Add a new prompt version.

        Sets the frozen ``promotion_score`` from the explicit
        ``promotion_score`` arg if provided, otherwise from ``score``.
        Once written, ``promotion_score`` is never modified — only
        ``update_score`` runs, and that only touches the running ``score``.
        """
        version = self.get_latest_version_number() + 1
        frozen = promotion_score if promotion_score is not None else score
        pv = PromptVersion(
            version=version,
            prompt=prompt,
            score=score,
            promotion_score=frozen,
            parent_version=parent_version,
            feedback_summary=feedback_summary,
        )
        path = self._version_path(version)
        data = pv.to_dict()
        if score is not None:
            data["score_count"] = 1
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "Saved prompt version %d (score=%s, promotion_score=%s)",
            version,
            score,
            frozen,
        )
        return pv

    def update_score(self, version: int, score: float) -> None:
        """Update the running observed score for an existing prompt version.

        Uses incremental averaging: new_avg = old_avg + (score - old_avg) / n.
        This is the running observation that lets the optimizer notice when
        a champion is degrading. It is intentionally separate from
        ``promotion_score``, which is frozen at promotion time and used for
        comparing champions across versions on a stable baseline.
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
        # promotion_score is frozen — never touched here.
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
