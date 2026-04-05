"""Tests for harness middleware."""

from evoagent.harness.middleware import (
    LoopDetectionMiddleware,
    check_output,
    is_similar_query,
)


def test_check_output_detects_error():
    issues = check_output("Error: API quota exceeded")
    assert any("error" in i.lower() for i in issues)


def test_check_output_accepts_good_report():
    report = (
        "# Report\n## Summary\nThis is a finding about the topic.\n"
        "## Sources\n- Source 1\n" + "x" * 500
    )
    issues = check_output(report, required_sections=["summary", "finding", "source"])
    assert issues == []


def test_check_output_detects_short():
    issues = check_output("Too short", min_length=500)
    assert any("short" in i for i in issues)


def test_check_output_custom_sections():
    issues = check_output(
        "Some content " * 100,
        required_sections=["recommendation", "risk"],
    )
    assert any("recommendation" in i for i in issues)


def test_is_similar_query():
    assert is_similar_query("quantum computing advances", "advances in quantum computing")
    assert not is_similar_query("quantum computing", "weather forecast today")


def test_loop_detection_tracks_queries():
    mw = LoopDetectionMiddleware(max_similar=2, max_total=5)
    assert mw.should_warn("quantum computing") is False
    assert mw.should_warn("quantum computing research") is False
    assert mw.should_warn("quantum computing advances") is True
