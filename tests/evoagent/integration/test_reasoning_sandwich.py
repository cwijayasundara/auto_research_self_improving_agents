"""Tests for ReasoningSandwichMiddleware."""

from evoagent.harness.middleware import ReasoningSandwichMiddleware


def test_planning_phase():
    mw = ReasoningSandwichMiddleware(planning_calls=2)
    assert mw.get_reasoning_effort() == "high"
    mw.increment_call()
    assert mw.get_reasoning_effort() == "high"


def test_implementation_phase():
    mw = ReasoningSandwichMiddleware(planning_calls=2)
    mw._call_count = 2
    assert mw.get_reasoning_effort() == "medium"


def test_verification_phase():
    mw = ReasoningSandwichMiddleware(planning_calls=2)
    mw._call_count = 5
    mw._in_verification = True
    assert mw.get_reasoning_effort() == "high"


def test_enter_verification():
    mw = ReasoningSandwichMiddleware(planning_calls=2)
    mw._call_count = 5
    assert mw.get_reasoning_effort() == "medium"
    mw.enter_verification()
    assert mw.get_reasoning_effort() == "high"


def test_custom_efforts():
    mw = ReasoningSandwichMiddleware(
        planning_effort="xhigh",
        implementation_effort="low",
        verification_effort="xhigh",
        planning_calls=1,
    )
    assert mw.get_reasoning_effort() == "xhigh"
    mw._call_count = 1
    assert mw.get_reasoning_effort() == "low"
    mw.enter_verification()
    assert mw.get_reasoning_effort() == "xhigh"
