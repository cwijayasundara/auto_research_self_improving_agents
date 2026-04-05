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

    def save(self, config: HarnessConfig, score: float | None = None) -> int:
        version = self.get_latest_version() + 1
        data = {
            "version": version,
            "config": config.to_dict(),
            "score": round(score, 4) if score is not None else None,
            "score_count": 1 if score is not None else 0,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        path = self._version_path(version)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved harness config version %d (score=%s)", version, score)
        return version

    def load_best(self) -> HarnessConfig:
        all_versions = self._load_all()
        if not all_versions:
            return HarnessConfig()
        scored = [v for v in all_versions if v.get("score") is not None]
        if scored:
            best = max(scored, key=lambda v: v["score"])
        else:
            best = all_versions[-1]
        return HarnessConfig.from_dict(best["config"])

    def update_score(self, version: int, score: float) -> None:
        """Update score using incremental averaging."""
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

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "Updated score for harness config version %d to %.4f (n=%d)",
            version,
            data["score"],
            data["score_count"],
        )
