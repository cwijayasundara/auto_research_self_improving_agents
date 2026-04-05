"""LangSmith trace fetcher."""

import logging
from pathlib import Path
from typing import Any

from langsmith import Client

from src.config.settings import Settings
from evoagent.tracing.trajectory import ToolCall, TrajectoryRecord

logger = logging.getLogger(__name__)


class TraceFetcher:
    """Fetches and parses traces from LangSmith."""

    def __init__(self, settings: Settings, cache_dir: Path | None = None) -> None:
        api_key = settings.resolved_api_key
        if not api_key:
            logger.warning("No LangSmith API key set. Trace fetching will be skipped.")
        self.client = Client(api_key=api_key or None)
        self.project_name = settings.langsmith_project
        self.cache_dir = cache_dir or (Path("traces"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_recent_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        """Fetch recent runs from LangSmith project."""
        runs = list(
            self.client.list_runs(
                project_name=self.project_name,
                limit=limit,
            )
        )
        return [self._run_to_dict(r) for r in runs]

    def fetch_run(self, run_id: str) -> dict[str, Any] | None:
        """Fetch a specific run by ID."""
        try:
            run = self.client.read_run(run_id)
            return self._run_to_dict(run)
        except Exception as exc:
            logger.error("Failed to fetch run %s: %s", run_id, exc)
            return None

    def _run_to_dict(self, run: Any) -> dict[str, Any]:
        """Convert a LangSmith run object to a serializable dict."""
        return {
            "run_id": str(run.id),
            "name": run.name or "",
            "status": run.status or "unknown",
            "inputs": run.inputs or {},
            "outputs": run.outputs or {},
            "start_time": str(run.start_time) if run.start_time else "",
            "end_time": str(run.end_time) if run.end_time else "",
            "total_tokens": run.total_tokens or 0,
            "feedback": {},
        }

    def parse_trajectory(self, run_data: dict[str, Any]) -> TrajectoryRecord:
        """Parse a raw run dict into a TrajectoryRecord model."""
        inputs = run_data.get("inputs", {})
        outputs = run_data.get("outputs", {})

        task = ""
        if isinstance(inputs, dict):
            task = inputs.get("input", inputs.get("task", str(inputs)))

        output = ""
        if isinstance(outputs, dict):
            raw_output = outputs.get("output", outputs.get("result", outputs))
            output = raw_output if isinstance(raw_output, str) else str(raw_output)

        return TrajectoryRecord(
            run_id=run_data["run_id"],
            task=task,
            output=output,
            tool_calls=[],
            total_tokens=run_data.get("total_tokens", 0),
            status=run_data.get("status", "unknown"),
        )

    def fetch_and_parse(self, limit: int = 10) -> list[TrajectoryRecord]:
        """Fetch recent runs and parse them into Trajectory models.

        Returns an empty list so the orchestrator uses its local
        trajectories, which have correct output and task data.
        LangSmith traces from deepagents have empty/misformatted
        outputs and mix in internal traces (prompt optimizer, skill
        creator), producing worse grading results.
        """
        logger.info("Skipping LangSmith trace fetch; using local trajectories")
        return []
