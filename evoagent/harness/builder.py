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
    harness_config: Any = None,
) -> list:
    """Build the default harness middleware stack.

    When *harness_config* (a ``HarnessConfig`` instance) is provided its
    values override the individual keyword defaults.

    Returns 6 middleware instances in recommended order.
    """
    return [
        SelfVerificationMiddleware(
            required_sections=required_sections or (harness_config.required_sections if harness_config else None),
            max_retries=harness_config.max_retries if harness_config else 2,
            min_length=harness_config.min_length if harness_config else 500,
            verify_against_task=harness_config.verify_against_task if harness_config else verify_against_task,
            llm=llm,
        ),
        ContextAssemblyMiddleware(
            skills_dir=skills_dir,
            detect_env=harness_config.detect_env if harness_config else detect_env,
            working_dir=working_dir,
        ),
        LoopDetectionMiddleware(
            max_similar=harness_config.max_similar if harness_config else 3,
            max_total=harness_config.max_total if harness_config else 12,
            max_file_edits=harness_config.max_file_edits if harness_config else 5,
            max_repeated_tools=harness_config.max_repeated_tools if harness_config else 4,
        ),
        TimeBudgetMiddleware(
            budget_seconds=harness_config.budget_seconds if harness_config else budget_seconds,
            warn_at=harness_config.warn_at if harness_config else [0.6, 0.85],
        ),
        ReasoningSandwichMiddleware(
            planning_effort=harness_config.planning_effort if harness_config else "high",
            implementation_effort=harness_config.implementation_effort if harness_config else "medium",
            verification_effort=harness_config.verification_effort if harness_config else "high",
            planning_calls=harness_config.planning_calls if harness_config else 2,
        ),
        TraceCaptureMiddleware(task=task, traces_dir=traces_dir),
    ]
