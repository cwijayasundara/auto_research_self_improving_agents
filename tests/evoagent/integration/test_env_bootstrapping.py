"""Integration tests for environment bootstrapping in ContextAssemblyMiddleware."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from evoagent.harness.middleware import detect_environment, format_environment_context


# ---------------------------------------------------------------------------
# detect_environment
# ---------------------------------------------------------------------------

class TestDetectEnvironment:
    def test_returns_expected_keys(self, tmp_path):
        result = detect_environment(tmp_path)
        assert set(result.keys()) == {"working_directory", "directory_listing", "available_tools"}

    def test_working_directory_is_string(self, tmp_path):
        result = detect_environment(tmp_path)
        assert isinstance(result["working_directory"], str)
        assert result["working_directory"] == str(tmp_path)

    def test_available_tools_is_list(self, tmp_path):
        result = detect_environment(tmp_path)
        assert isinstance(result["available_tools"], list)

    def test_finds_python3_in_available_tools(self, tmp_path):
        result = detect_environment(tmp_path)
        # python3 should be present in any environment running these tests
        assert "python3" in result["available_tools"] or "python" in result["available_tools"]

    def test_directory_listing_includes_files(self, tmp_path):
        (tmp_path / "hello.txt").write_text("hi")
        result = detect_environment(tmp_path)
        assert "hello.txt" in result["directory_listing"]

    def test_directory_listing_includes_dirs_with_slash(self, tmp_path):
        subdir = tmp_path / "mysubdir"
        subdir.mkdir()
        result = detect_environment(tmp_path)
        assert "mysubdir/" in result["directory_listing"]

    def test_listing_skips_dotfiles(self, tmp_path):
        (tmp_path / ".hidden").write_text("secret")
        (tmp_path / "visible.txt").write_text("shown")
        result = detect_environment(tmp_path)
        assert ".hidden" not in result["directory_listing"]
        assert "visible.txt" in result["directory_listing"]

    def test_listing_caps_at_max_entries(self, tmp_path):
        for i in range(30):
            (tmp_path / f"file_{i:02d}.txt").write_text("x")
        result = detect_environment(tmp_path, max_entries=10)
        lines = [line for line in result["directory_listing"].splitlines() if line.strip()]
        assert len(lines) <= 10

    def test_listing_respects_max_depth(self, tmp_path):
        deep = tmp_path / "level1" / "level2" / "level3"
        deep.mkdir(parents=True)
        (deep / "deep_file.txt").write_text("deep")
        result = detect_environment(tmp_path, max_depth=2)
        # deep_file.txt is 3 levels down — should not appear
        assert "deep_file.txt" not in result["directory_listing"]

    def test_empty_directory_returns_empty_sentinel(self, tmp_path):
        result = detect_environment(tmp_path)
        assert result["directory_listing"] == "(empty)"

    def test_accepts_path_object(self, tmp_path):
        result = detect_environment(Path(tmp_path))
        assert result["working_directory"] == str(tmp_path)

    def test_accepts_string_path(self, tmp_path):
        result = detect_environment(str(tmp_path))
        assert result["working_directory"] == str(tmp_path)


# ---------------------------------------------------------------------------
# format_environment_context
# ---------------------------------------------------------------------------

class TestFormatEnvironmentContext:
    def _make_env(self, **kwargs):
        base = {
            "working_directory": "/some/path",
            "directory_listing": "file.txt\nsubdir/",
            "available_tools": ["python3", "git"],
        }
        base.update(kwargs)
        return base

    def test_has_system_environment_header(self):
        result = format_environment_context(self._make_env())
        assert "## System Environment" in result

    def test_includes_working_directory(self):
        result = format_environment_context(self._make_env(working_directory="/my/project"))
        assert "/my/project" in result

    def test_includes_available_tools(self):
        result = format_environment_context(self._make_env(available_tools=["python3", "git", "curl"]))
        assert "python3" in result
        assert "git" in result
        assert "curl" in result

    def test_includes_directory_listing(self):
        result = format_environment_context(self._make_env(directory_listing="main.py\nREADME.md"))
        assert "main.py" in result
        assert "README.md" in result

    def test_no_tools_detected_shows_none(self):
        result = format_environment_context(self._make_env(available_tools=[]))
        assert "none detected" in result

    def test_output_is_readable_markdown(self):
        env = detect_environment(Path("."))
        result = format_environment_context(env)
        assert isinstance(result, str)
        assert len(result) > 20
        # Should have markdown-style header
        assert result.startswith("## ")
