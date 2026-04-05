"""Tests for LoopDetectionMiddleware file-edit and repeated-tool tracking."""

from unittest.mock import MagicMock

import pytest

from evoagent.harness.middleware import LoopDetectionMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(tool_name: str, args: dict) -> MagicMock:
    req = MagicMock()
    req.name = tool_name
    req.args = args
    return req


def _make_result(content: str = "ok", tool_call_id: str = "tc1") -> MagicMock:
    result = MagicMock()
    result.content = content
    result.tool_call_id = tool_call_id
    result.__dict__["content"] = content
    result.__dict__["tool_call_id"] = tool_call_id
    return result


# ---------------------------------------------------------------------------
# File-edit tracking tests
# ---------------------------------------------------------------------------


class TestShouldWarnFileEdit:
    def test_triggers_at_threshold(self):
        mw = LoopDetectionMiddleware(max_file_edits=3)
        assert mw.should_warn_file_edit("foo.py") is False  # count=1
        assert mw.should_warn_file_edit("foo.py") is False  # count=2
        assert mw.should_warn_file_edit("foo.py") is True   # count=3 → trigger

    def test_different_files_do_not_trigger_each_other(self):
        mw = LoopDetectionMiddleware(max_file_edits=3)
        mw.should_warn_file_edit("foo.py")
        mw.should_warn_file_edit("foo.py")
        # bar.py starts at 0 — should not trigger
        assert mw.should_warn_file_edit("bar.py") is False

    def test_warning_fires_only_once_per_file(self):
        mw = LoopDetectionMiddleware(max_file_edits=2)
        mw.should_warn_file_edit("foo.py")      # count=1
        assert mw.should_warn_file_edit("foo.py") is True   # count=2, first warn
        assert mw.should_warn_file_edit("foo.py") is False  # count=3, already warned
        assert mw.should_warn_file_edit("foo.py") is False  # count=4, still no double-warn

    def test_count_tracked_correctly(self):
        mw = LoopDetectionMiddleware(max_file_edits=5)
        for _ in range(4):
            mw.should_warn_file_edit("main.py")
        assert mw._file_edit_counts["main.py"] == 4
        assert mw.should_warn_file_edit("main.py") is True
        assert mw._file_edit_counts["main.py"] == 5


# ---------------------------------------------------------------------------
# Repeated-tool tracking tests
# ---------------------------------------------------------------------------


class TestShouldWarnRepeatedTool:
    def test_triggers_at_threshold(self):
        mw = LoopDetectionMiddleware(max_repeated_tools=3)
        assert mw.should_warn_repeated_tool("web_search", "climate change research") is False
        assert mw.should_warn_repeated_tool("web_search", "climate change advances research") is False
        # 3rd similar call → trigger
        assert mw.should_warn_repeated_tool("web_search", "climate change research advances") is True

    def test_different_tools_do_not_cross_trigger(self):
        mw = LoopDetectionMiddleware(max_repeated_tools=3)
        mw.should_warn_repeated_tool("web_search", "climate change")
        mw.should_warn_repeated_tool("web_search", "climate change research")
        # read_file with unrelated args should not trigger
        assert mw.should_warn_repeated_tool("read_file", "path=/etc/hosts") is False

    def test_warning_fires_only_once(self):
        mw = LoopDetectionMiddleware(max_repeated_tools=2)
        mw.should_warn_repeated_tool("web_search", "climate change research")
        assert mw.should_warn_repeated_tool("web_search", "climate change research data") is True
        # subsequent similar calls should NOT warn again
        assert mw.should_warn_repeated_tool("web_search", "climate change research findings") is False

    def test_dissimilar_args_do_not_trigger(self):
        mw = LoopDetectionMiddleware(max_repeated_tools=3)
        mw.should_warn_repeated_tool("web_search", "quantum computing")
        mw.should_warn_repeated_tool("web_search", "machine learning")
        assert mw.should_warn_repeated_tool("web_search", "weather today") is False


# ---------------------------------------------------------------------------
# wrap_tool_call integration tests
# ---------------------------------------------------------------------------


class TestWrapToolCall:
    def test_file_edit_warning_appended(self):
        mw = LoopDetectionMiddleware(max_file_edits=2, max_repeated_tools=10)
        request = _make_request("edit_file", {"file_path": "app.py"})
        handler = MagicMock(return_value=_make_result("edited"))

        # First two calls — second should trigger file-edit warning
        mw.wrap_tool_call(request, handler)
        result = mw.wrap_tool_call(request, handler)

        assert "_loop_detection_warnings" in result.__dict__
        warnings = result.__dict__["_loop_detection_warnings"]
        assert len(warnings) >= 1
        assert any("LOOP WARNING" in w.content for w in warnings)

    def test_no_warning_below_threshold(self):
        mw = LoopDetectionMiddleware(max_file_edits=5, max_repeated_tools=10)
        request = _make_request("edit_file", {"file_path": "app.py"})
        handler = MagicMock(return_value=_make_result("edited"))

        result = mw.wrap_tool_call(request, handler)
        assert "_loop_detection_warnings" not in result.__dict__

    def test_non_edit_tool_not_tracked_for_file_edits(self):
        mw = LoopDetectionMiddleware(max_file_edits=2, max_repeated_tools=10)
        request = _make_request("read_file", {"file_path": "app.py"})
        handler = MagicMock(return_value=_make_result("content"))

        mw.wrap_tool_call(request, handler)
        result = mw.wrap_tool_call(request, handler)

        # file-edit warning should NOT fire for read_file
        warnings = result.__dict__.get("_loop_detection_warnings", [])
        assert not any("LOOP WARNING" in (w.content if hasattr(w, "content") else "") for w in warnings)

    def test_path_key_extracted(self):
        mw = LoopDetectionMiddleware(max_file_edits=2, max_repeated_tools=10)
        # Uses 'path' instead of 'file_path'
        request = _make_request("write_file", {"path": "output.txt"})
        handler = MagicMock(return_value=_make_result("written"))

        mw.wrap_tool_call(request, handler)
        result = mw.wrap_tool_call(request, handler)

        assert "_loop_detection_warnings" in result.__dict__

    def test_filename_key_extracted(self):
        mw = LoopDetectionMiddleware(max_file_edits=2, max_repeated_tools=10)
        request = _make_request("create_file", {"filename": "notes.md"})
        handler = MagicMock(return_value=_make_result("created"))

        mw.wrap_tool_call(request, handler)
        result = mw.wrap_tool_call(request, handler)

        assert "_loop_detection_warnings" in result.__dict__
