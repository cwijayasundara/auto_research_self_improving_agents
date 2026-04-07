"""Tests for the daemon's _load_holdout_tasks helper.

Pins:
1. Loads tasks from settings.tasks_file_path and reuses _split_tasks (seed 42)
2. Returns empty list cleanly when file is missing
3. Returns empty list cleanly when file is malformed
4. Same input → same holdout slice across cycles (deterministic)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

from src.evolution.daemon import _load_holdout_tasks


def _settings_for(tmp_path: Path, tasks_filename: str | None = None):
    """Build a Settings stub that points tasks_file_path at tmp_path."""
    settings = MagicMock()
    if tasks_filename is not None:
        settings.tasks_file_path = tmp_path / tasks_filename
    else:
        settings.tasks_file_path = tmp_path / "nonexistent.json"
    return settings


class TestLoadHoldoutTasks:
    def test_returns_empty_when_file_missing(self, tmp_path: Path):
        settings = _settings_for(tmp_path, None)
        assert _load_holdout_tasks(settings) == []

    def test_returns_empty_when_file_is_not_json_list(self, tmp_path: Path):
        bad = tmp_path / "tasks.json"
        bad.write_text('{"not": "a list"}')
        settings = _settings_for(tmp_path, "tasks.json")
        assert _load_holdout_tasks(settings) == []

    def test_returns_empty_when_file_is_invalid_json(self, tmp_path: Path):
        bad = tmp_path / "tasks.json"
        bad.write_text("not valid json {{{")
        settings = _settings_for(tmp_path, "tasks.json")
        assert _load_holdout_tasks(settings) == []

    def test_returns_empty_when_too_few_tasks(self, tmp_path: Path):
        """_split_tasks returns ([], []) when len(tasks) <= MIN_TRAINING + MIN_HOLDOUT."""
        f = tmp_path / "tasks.json"
        f.write_text(json.dumps(["only", "two"]))
        settings = _settings_for(tmp_path, "tasks.json")
        # 2 tasks <= 2+1, so split returns all training, no holdout
        assert _load_holdout_tasks(settings) == []

    def test_loads_holdout_slice_from_real_task_list(self, tmp_path: Path):
        f = tmp_path / "tasks.json"
        tasks = [f"task {i}" for i in range(10)]
        f.write_text(json.dumps(tasks))
        settings = _settings_for(tmp_path, "tasks.json")

        holdout = _load_holdout_tasks(settings)
        assert len(holdout) > 0
        assert len(holdout) < len(tasks)  # only a fraction is held out
        # Every held-out task is in the original list
        assert all(t in tasks for t in holdout)

    def test_holdout_slice_is_deterministic(self, tmp_path: Path):
        """Two consecutive loads of the same file produce the same holdout
        slice — this is what makes the gate stable across cycles."""
        f = tmp_path / "tasks.json"
        tasks = [f"task {i}" for i in range(10)]
        f.write_text(json.dumps(tasks))
        settings = _settings_for(tmp_path, "tasks.json")

        first = _load_holdout_tasks(settings)
        second = _load_holdout_tasks(settings)
        assert first == second
