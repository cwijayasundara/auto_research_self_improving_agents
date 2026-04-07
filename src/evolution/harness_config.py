"""Versioned harness configuration store.

Stores tunable middleware parameters as versioned JSON files.
Supports scoring, incremental averaging, and best-version retrieval.
"""

import json
import logging
from dataclasses import dataclass, field, fields, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HarnessConfig:
    """All tunable middleware parameters."""

    # SelfVerification
    max_retries: int = 2
    min_length: int = 500
    verify_against_task: bool = False
    required_sections: list[str] = field(
        default_factory=lambda: ["summary", "finding", "source"]
    )

    # Custom completion checks (evaluated by LLM)
    completion_checks: list[str] = field(default_factory=list)

    # LoopDetection
    max_similar: int = 3
    max_total: int = 12
    max_file_edits: int = 5
    max_repeated_tools: int = 4

    # TimeBudget
    budget_seconds: int = 300
    warn_at: list[float] = field(default_factory=lambda: [0.6, 0.85])

    # ReasoningSandwich
    planning_effort: str = "high"
    implementation_effort: str = "medium"
    verification_effort: str = "high"
    planning_calls: int = 2

    # ContextAssembly
    detect_env: bool = True

    # Search tool
    search_max_retries: int = 1
    search_retry_delay: int = 2
    search_max_results: int = 3
    search_depth: str = "basic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HarnessConfig":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


class HarnessConfigStore:
    """File-based versioned harness config store."""

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = config_dir
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _version_path(self, version: int) -> Path:
        return self.config_dir / f"v{version:04d}.json"

    def _load_all(self) -> list[dict[str, Any]]:
        results = []
        for path in sorted(self.config_dir.glob("v*.json")):
            with open(path) as f:
                results.append(json.load(f))
        return results

    def get_latest_version(self) -> int:
        all_versions = self._load_all()
        if not all_versions:
            return 0
        return all_versions[-1]["version"]

    def save(
        self,
        config: HarnessConfig,
        score: float | None = None,
        promotion_score: float | None = None,
    ) -> int:
        """Save a new harness config version.

        Sets the frozen ``promotion_score`` from the explicit
        ``promotion_score`` arg if provided, otherwise from ``score``.
        Once written, ``promotion_score`` is never modified by
        ``update_score`` — it is the stable baseline used by the ratchet
        to compare champions across versions.
        """
        version = self.get_latest_version() + 1
        frozen = promotion_score if promotion_score is not None else score
        data = {
            "version": version,
            "config": config.to_dict(),
            "score": round(score, 4) if score is not None else None,
            "score_count": 1 if score is not None else 0,
            "promotion_score": round(frozen, 4) if frozen is not None else None,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        path = self._version_path(version)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "Saved harness config version %d (score=%s, promotion_score=%s)",
            version,
            score,
            frozen,
        )
        return version

    def load_best(self) -> HarnessConfig:
        """Return the current champion harness config — the LATEST version.

        This is the ratchet: promotions only move forward. The active
        config is whichever was promoted most recently. We do NOT select
        by max-score, because doing so would let observed-score drift
        silently roll the active config back to an older version.
        """
        all_versions = self._load_all()
        if not all_versions:
            return HarnessConfig()
        return HarnessConfig.from_dict(all_versions[-1]["config"])

    def update_score(self, version: int, score: float) -> None:
        """Update the running observed score for a harness config version.

        Uses incremental averaging. Touches ONLY ``score``, never
        ``promotion_score`` — the latter is frozen at promotion time and
        is the stable baseline for the ratchet.
        """
        path = self._version_path(version)
        if not path.exists():
            logger.warning("Cannot update score: version %d not found", version)
            return

        with open(path) as f:
            data = json.load(f)

        n = data.get("score_count", 1 if data.get("score") is not None else 0)
        old_score = data.get("score")

        if n == 0 or old_score is None:
            data["score"] = round(score, 4)
            data["score_count"] = 1
        else:
            n += 1
            new_score = old_score + (score - old_score) / n
            data["score"] = round(new_score, 4)
            data["score_count"] = n

        # promotion_score is frozen — never touched here.
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "Updated score for harness config version %d to %.4f (n=%d)",
            version,
            data["score"],
            data["score_count"],
        )
