"""Evolution state persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def persist_evolution_state(
    state_dir: Path,
    cycle: int,
    metrics: dict[str, Any],
    analyses: list[dict[str, Any]],
    plateau_reason: str | None = None,
) -> None:
    """Write evolution state to disk."""
    state_dir.mkdir(parents=True, exist_ok=True)

    (state_dir / f"cycle_{cycle}_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str)
    )
    (state_dir / f"cycle_{cycle}_analyses.json").write_text(
        json.dumps(analyses, indent=2, default=str)
    )

    if plateau_reason:
        (state_dir / "plateau_report.md").write_text(
            f"# Plateau Report\n\nCycle: {cycle}\nReason: {plateau_reason}\n"
        )

    logger.info("Persisted evolution state for cycle %d", cycle)
