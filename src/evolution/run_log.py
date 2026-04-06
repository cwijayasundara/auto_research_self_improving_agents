"""Append-only run log for single-run results.

Each user interaction appends a structured entry. The background evolution
daemon reads unprocessed entries and triggers optimization when enough
signal has accumulated.
"""

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

# File-level lock for safe concurrent appends
_write_lock = threading.Lock()


class RunLogEntry:
    """A single run result stored in the log."""

    def __init__(
        self,
        run_id: str,
        task: str,
        output: str,
        classification: str,
        average_score: float,
        grader_results: list[dict[str, Any]],
        prompt_version: int,
        harness_config_version: int = 0,
        trace_path: str = "",
        timestamp: str | None = None,
        processed: bool = False,
    ) -> None:
        self.run_id = run_id
        self.task = task
        self.output = output
        self.classification = classification
        self.average_score = average_score
        self.grader_results = grader_results
        self.prompt_version = prompt_version
        self.harness_config_version = harness_config_version
        self.trace_path = trace_path
        self.timestamp = timestamp or datetime.now(UTC).isoformat()
        self.processed = processed

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "output": self.output,
            "classification": self.classification,
            "average_score": self.average_score,
            "grader_results": self.grader_results,
            "prompt_version": self.prompt_version,
            "harness_config_version": self.harness_config_version,
            "trace_path": self.trace_path,
            "timestamp": self.timestamp,
            "processed": self.processed,
        }


class RunLog:
    """Append-only JSON-lines run log."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: RunLogEntry) -> None:
        """Append a run result to the log (thread-safe)."""
        with _write_lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        logger.info("Run logged: %s (score=%.3f)", entry.run_id[:8], entry.average_score)

    def read_all(self) -> list[RunLogEntry]:
        """Read all entries from the log."""
        if not self.log_path.exists():
            return []
        entries: list[RunLogEntry] = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(RunLogEntry(**data))
                except (json.JSONDecodeError, TypeError) as exc:
                    logger.warning("Skipping malformed run log entry: %s", exc)
        return entries

    def read_unprocessed(self) -> list[RunLogEntry]:
        """Read only entries not yet processed by the evolution daemon."""
        return [e for e in self.read_all() if not e.processed]

    def mark_processed(self, run_ids: set[str]) -> None:
        """Mark entries as processed by rewriting the log."""
        if not run_ids or not self.log_path.exists():
            return
        with _write_lock:
            entries = self.read_all()
            with open(self.log_path, "w") as f:
                for entry in entries:
                    if entry.run_id in run_ids:
                        entry.processed = True
                    f.write(json.dumps(entry.to_dict()) + "\n")

    def count_unprocessed(self) -> int:
        """Count unprocessed entries without loading all data."""
        if not self.log_path.exists():
            return 0
        count = 0
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if not data.get("processed", False):
                        count += 1
                except json.JSONDecodeError:
                    pass
        return count
