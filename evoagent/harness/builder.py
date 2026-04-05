"""Factory for building default middleware stacks."""

from __future__ import annotations

from pathlib import Path

from evoagent.harness.middleware import (
    ContextAssemblyMiddleware,
    LoopDetectionMiddleware,
    SelfVerificationMiddleware,
    TraceCaptureMiddleware,
)


def default_middleware_stack(
    task: str = "",
    skills_dir: Path | None = None,
    traces_dir: Path | None = None,
    required_sections: list[str] | None = None,
) -> list:
    return [
        SelfVerificationMiddleware(required_sections=required_sections),
        ContextAssemblyMiddleware(skills_dir=skills_dir),
        LoopDetectionMiddleware(),
        TraceCaptureMiddleware(task=task, traces_dir=traces_dir),
    ]
