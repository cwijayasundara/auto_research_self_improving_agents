"""Tests for updated default_middleware_stack."""

from evoagent.harness.builder import default_middleware_stack
from evoagent.harness.middleware import (
    ContextAssemblyMiddleware,
    LoopDetectionMiddleware,
    SelfVerificationMiddleware,
    TimeBudgetMiddleware,
    TraceCaptureMiddleware,
    ReasoningSandwichMiddleware,
)


def test_default_stack_has_6_middleware():
    stack = default_middleware_stack()
    assert len(stack) == 6


def test_default_stack_types():
    stack = default_middleware_stack()
    types = [type(m).__name__ for m in stack]
    assert "SelfVerificationMiddleware" in types
    assert "ContextAssemblyMiddleware" in types
    assert "LoopDetectionMiddleware" in types
    assert "TimeBudgetMiddleware" in types
    assert "ReasoningSandwichMiddleware" in types
    assert "TraceCaptureMiddleware" in types


def test_stack_with_env_detection():
    stack = default_middleware_stack(detect_env=True)
    ctx = [m for m in stack if isinstance(m, ContextAssemblyMiddleware)]
    assert len(ctx) == 1
    assert ctx[0]._detect_env is True


def test_stack_with_time_budget():
    stack = default_middleware_stack(budget_seconds=120)
    tb = [m for m in stack if isinstance(m, TimeBudgetMiddleware)]
    assert len(tb) == 1
    assert tb[0]._budget == 120


def test_all_exports_importable():
    from evoagent.harness import (
        ContextAssemblyMiddleware,
        LoopDetectionMiddleware,
        ReasoningSandwichMiddleware,
        SelfVerificationMiddleware,
        TimeBudgetMiddleware,
        TraceCaptureMiddleware,
        default_middleware_stack,
    )
    assert callable(default_middleware_stack)
