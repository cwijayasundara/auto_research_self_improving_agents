"""Factory for building default middleware stacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evoagent.harness.middleware import (
    ContextAssemblyMiddleware,
    LoopDetectionMiddleware,
    ReasoningSandwichMiddleware,
    SelfVerificationMiddleware,
    TimeBudgetMiddleware,
    TraceCaptureMiddleware,
)


def default_middleware_stack(
    task: str = "",
    skills_dir: Path | None = None,
    traces_dir: Path | None = None,
    required_sections: list[str] | None = None,
    detect_env: bool = False,
    working_dir: Path | None = None,
    budget_seconds: int = 300,
    verify_against_task: bool = False,
    llm: Any = None,
) -> list:
    """Build the default harness middleware stack.

    Returns 6 middleware instances in recommended order.
    """
    return [
        SelfVerificationMiddleware(
            required_sections=required_sections,
            verify_against_task=verify_against_task,
            llm=llm,
        ),
        ContextAssemblyMiddleware(
            skills_dir=skills_dir,
            detect_env=detect_env,
            working_dir=working_dir,
        ),
        LoopDetectionMiddleware(),
        TimeBudgetMiddleware(budget_seconds=budget_seconds),
        ReasoningSandwichMiddleware(),
        TraceCaptureMiddleware(task=task, traces_dir=traces_dir),
    ]
