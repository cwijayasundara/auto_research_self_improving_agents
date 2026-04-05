"""Tests for TimeBudgetMiddleware."""

import time

import pytest

from evoagent.harness.middleware import TimeBudgetMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeToolMessage:
    """Minimal stand-in for a tool result with string content."""

    def __init__(self, content: str, tool_call_id: str = "tc-1") -> None:
        self.content = content
        self.tool_call_id = tool_call_id


def _make_mw(**kwargs) -> TimeBudgetMiddleware:
    mw = TimeBudgetMiddleware(**kwargs)
    mw.start()
    return mw


def _rewind(mw: TimeBudgetMiddleware, fraction: float) -> None:
    """Shift _start_time so that elapsed_fraction() returns *fraction*."""
    mw._start_time = time.monotonic() - fraction * mw._budget


# ---------------------------------------------------------------------------
# elapsed_fraction
# ---------------------------------------------------------------------------

def test_elapsed_fraction_before_start():
    mw = TimeBudgetMiddleware()
    assert mw.elapsed_fraction() == 0.0


def test_elapsed_fraction_correct_value():
    mw = TimeBudgetMiddleware(budget_seconds=100)
    mw.start()
    _rewind(mw, 0.5)
    frac = mw.elapsed_fraction()
    assert 0.48 <= frac <= 0.52, f"Expected ~0.5, got {frac}"


# ---------------------------------------------------------------------------
# No warning before threshold
# ---------------------------------------------------------------------------

def test_no_warning_before_any_threshold():
    mw = TimeBudgetMiddleware(budget_seconds=100)
    mw.start()
    _rewind(mw, 0.3)  # below 0.6
    assert mw.get_warning() is None


# ---------------------------------------------------------------------------
# Warning at 60 %
# ---------------------------------------------------------------------------

def test_warning_fires_at_60_percent():
    mw = TimeBudgetMiddleware(budget_seconds=100)
    mw.start()
    _rewind(mw, 0.65)  # between 0.6 and 0.85
    warning = mw.get_warning()
    assert warning == "Start synthesizing"


def test_warning_at_60_does_not_contain_urgent():
    mw = TimeBudgetMiddleware(budget_seconds=100)
    mw.start()
    _rewind(mw, 0.65)
    assert "URGENT" not in (mw.get_warning() or "")


# ---------------------------------------------------------------------------
# Warning at 85 % — URGENT
# ---------------------------------------------------------------------------

def test_urgent_warning_fires_at_85_percent():
    mw = TimeBudgetMiddleware(budget_seconds=100)
    mw.start()
    _rewind(mw, 0.90)  # above 0.85
    warning = mw.get_warning()
    assert warning is not None
    assert "URGENT" in warning


def test_urgent_message_content():
    mw = TimeBudgetMiddleware(budget_seconds=100)
    mw.start()
    _rewind(mw, 0.90)
    assert mw.get_warning() == "URGENT: stop all searches, write report NOW"


# ---------------------------------------------------------------------------
# Fires only once per threshold
# ---------------------------------------------------------------------------

def test_warning_fires_only_once_at_60():
    mw = TimeBudgetMiddleware(budget_seconds=100)
    mw.start()
    _rewind(mw, 0.65)
    first = mw.get_warning()
    second = mw.get_warning()
    assert first == "Start synthesizing"
    assert second is None


def test_warning_fires_only_once_at_85():
    """At 90%, the 0.85 threshold fires first (URGENT), then the 0.6 threshold
    fires on the next call (both crossed), and only then does None come back."""
    mw = TimeBudgetMiddleware(budget_seconds=100)
    mw.start()
    _rewind(mw, 0.90)
    first = mw.get_warning()
    assert first is not None and "URGENT" in first
    # 0.6 threshold is also crossed and not yet fired; it will fire now
    second = mw.get_warning()
    assert second == "Start synthesizing"
    # Both thresholds consumed — no more warnings
    third = mw.get_warning()
    assert third is None


def test_60_and_85_each_fire_once():
    """Simulate crossing 60 % then crossing 85 %."""
    mw = TimeBudgetMiddleware(budget_seconds=100)
    mw.start()

    _rewind(mw, 0.65)
    w1 = mw.get_warning()
    assert w1 == "Start synthesizing"

    _rewind(mw, 0.90)
    w2 = mw.get_warning()
    assert w2 is not None and "URGENT" in w2

    w3 = mw.get_warning()
    assert w3 is None


# ---------------------------------------------------------------------------
# wrap_tool_call
# ---------------------------------------------------------------------------

def test_wrap_tool_call_no_warning_returns_result_unchanged():
    mw = TimeBudgetMiddleware(budget_seconds=100)
    mw.start()
    _rewind(mw, 0.3)

    msg = _FakeToolMessage("search results")
    result = mw.wrap_tool_call(None, lambda _: msg)
    assert result is msg
    assert not hasattr(result, "_time_budget_warning")


def test_wrap_tool_call_appends_warning_when_threshold_crossed():
    mw = TimeBudgetMiddleware(budget_seconds=100)
    mw.start()
    _rewind(mw, 0.65)

    msg = _FakeToolMessage("search results")
    result = mw.wrap_tool_call(None, lambda _: msg)
    assert hasattr(result, "_time_budget_warning")
    warning_msg = result._time_budget_warning
    assert "synthesiz" in warning_msg.content.lower()


def test_wrap_tool_call_non_string_content_no_warning_attached():
    mw = TimeBudgetMiddleware(budget_seconds=100)
    mw.start()
    _rewind(mw, 0.65)

    class _NonStringResult:
        content = 42  # not a string

    result = mw.wrap_tool_call(None, lambda _: _NonStringResult())
    assert not hasattr(result, "_time_budget_warning")


# ---------------------------------------------------------------------------
# before_agent auto-start
# ---------------------------------------------------------------------------

def test_before_agent_auto_starts_timer():
    mw = TimeBudgetMiddleware(budget_seconds=100)
    assert mw._start_time == 0.0
    mw.before_agent({}, None)
    assert mw._start_time != 0.0


def test_before_agent_does_not_reset_existing_timer():
    mw = TimeBudgetMiddleware(budget_seconds=100)
    mw.start()
    original = mw._start_time
    time.sleep(0.01)
    mw.before_agent({}, None)
    assert mw._start_time == original
