"""LangSmith trace fetcher.

Fetches rich execution traces from LangSmith including child runs
(tool calls, model reasoning, errors). Used by the prompt optimizer
and harness optimizer for counterfactual diagnosis.

Falls back to local trace files when LangSmith is unavailable.
"""

import json
import logging
from pathlib import Path
from typing import Any

from src.config.settings import Settings
logger = logging.getLogger(__name__)

# Max chars per child run output to keep trace data manageable
_MAX_OUTPUT_CHARS = 500
_MAX_CHILD_RUNS = 30


class TraceFetcher:
    """Fetches and formats traces from LangSmith for optimization."""

    def __init__(self, settings: Settings, cache_dir: Path | None = None) -> None:
        self._api_key = settings.resolved_api_key
        self._project = settings.langsmith_project
        self._cache_dir = cache_dir or settings.traces_path
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    def _get_client(self):
        """Lazy-init LangSmith client."""
        if self._client is None and self._api_key:
            try:
                from langsmith import Client
                self._client = Client(api_key=self._api_key)
            except Exception as exc:
                logger.warning("Failed to create LangSmith client: %s", exc)
        return self._client

    def fetch_recent_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        """Fetch recent root runs from LangSmith project."""
        client = self._get_client()
        if not client:
            return []
        try:
            runs = list(
                client.list_runs(
                    project_name=self._project,
                    is_root=True,
                    limit=limit,
                )
            )
            return [self._run_to_dict(r) for r in runs]
        except Exception as exc:
            logger.warning("LangSmith fetch failed: %s", exc)
            return []

    def fetch_rich_trace(self, run_id: str) -> str:
        """Fetch a full trace tree from LangSmith and format as diagnostic text.

        Uses list_runs with trace_id to get ALL descendant runs (not just
        direct children), giving full visibility into tool calls, LLM
        reasoning, and errors at every depth.

        Returns empty string if LangSmith is unavailable.
        """
        client = self._get_client()
        if not client:
            return ""
        try:
            # Get root run for metadata
            root = client.read_run(run_id)
            # Get ALL descendant runs (tool calls, LLM calls, sub-chains)
            all_runs = list(client.list_runs(
                trace_id=root.trace_id,
                limit=_MAX_CHILD_RUNS + 1,
            ))
            return self._format_flat_trace(root, all_runs)
        except Exception as exc:
            logger.debug("Failed to fetch trace %s: %s", run_id, exc)
            return ""

    def fetch_rich_trace_for_task(self, task: str, max_results: int = 3) -> str:
        """Find recent LangSmith runs matching a task and return rich traces.

        Matches by comparing the first 100 chars of the task text against
        run inputs. Returns formatted trace data for all matches.
        """
        client = self._get_client()
        if not client:
            return ""

        try:
            task_prefix = task[:100]
            runs = list(
                client.list_runs(
                    project_name=self._project,
                    is_root=True,
                    limit=20,
                )
            )

            matches: list[str] = []
            for run in runs:
                run_input = ""
                if isinstance(run.inputs, dict):
                    # deepagents puts the task in messages[0].content or input
                    messages = run.inputs.get("messages", [])
                    if messages and isinstance(messages, list):
                        first = messages[0]
                        if isinstance(first, dict):
                            run_input = first.get("content", "")
                        elif hasattr(first, "content"):
                            run_input = first.content or ""
                    if not run_input:
                        run_input = str(run.inputs.get("input", ""))

                if run_input[:100] == task_prefix:
                    # Fetch full trace with all descendants
                    trace_text = self.fetch_rich_trace(str(run.id))
                    if trace_text:
                        matches.append(trace_text)
                    if len(matches) >= max_results:
                        break

            return "\n\n---\n\n".join(matches)
        except Exception as exc:
            logger.warning("LangSmith task trace fetch failed: %s", exc)
            return ""

    def fetch_traces_for_entries(self, entries: list, max_traces: int = 5) -> str:
        """Fetch rich LangSmith traces for a list of run log entries.

        Tries to match entries to recent LangSmith runs by task text.
        Returns combined trace text for all matches.
        Falls back to local traces if LangSmith unavailable.
        """
        client = self._get_client()

        if client:
            # Try LangSmith first
            trace_parts: list[str] = []
            for entry in entries[:max_traces]:
                task = getattr(entry, "task", "")
                if not task:
                    continue
                trace = self.fetch_rich_trace_for_task(task, max_results=1)
                if trace:
                    trace_parts.append(f"### Task: {task[:60]}\n{trace}")
            if trace_parts:
                logger.info(
                    "Loaded %d rich traces from LangSmith", len(trace_parts)
                )
                return "\n\n".join(trace_parts)

        # Fall back to local traces
        trace_parts = []
        for entry in entries[:max_traces]:
            trace_path = getattr(entry, "trace_path", "")
            if trace_path:
                local = _load_local_trace(trace_path)
                if local:
                    task = getattr(entry, "task", "?")
                    trace_parts.append(f"### Task: {task[:60]}\n{local}")

        if trace_parts:
            logger.info("Loaded %d local traces (LangSmith unavailable)", len(trace_parts))
        return "\n\n".join(trace_parts)

    def _format_flat_trace(self, root, all_runs: list) -> str:
        """Format all runs in a trace as flat diagnostic text.

        Sorts by start_time so the trace reads chronologically.
        Focuses on tool and llm runs which have the richest diagnostic data.
        """
        parts: list[str] = []

        # Root summary
        duration = ""
        if root.start_time and root.end_time:
            delta = (root.end_time - root.start_time).total_seconds()
            duration = f" ({delta:.1f}s)"
        parts.append(
            f"Root: {root.name} [{root.status or '?'}]{duration} "
            f"tokens={root.total_tokens or 0}"
        )
        if root.error:
            parts.append(f"ROOT ERROR: {root.error[:300]}")

        # Sort descendants by start_time, skip root.
        # Filter to tool and llm runs (skip orchestrator chain noise)
        descendants = [
            r for r in all_runs
            if str(r.id) != str(root.id)
            and r.run_type in ("tool", "llm", "chain")
        ]
        descendants.sort(key=lambda r: r.start_time or root.start_time)

        for i, child in enumerate(descendants[:_MAX_CHILD_RUNS]):
            run_type = child.run_type or "unknown"
            name = child.name or "?"

            if run_type == "tool":
                args = _truncate(str(child.inputs or {}), 200)
                output = _truncate(str(child.outputs or ""), _MAX_OUTPUT_CHARS)
                parts.append(f"  TOOL: {name}")
                parts.append(f"    args: {args}")
                if child.error:
                    parts.append(f"    ERROR: {child.error[:200]}")
                elif output:
                    parts.append(f"    result: {output}")

            elif run_type == "llm":
                child_tokens = child.total_tokens or 0
                # Extract tool call decisions from output
                output_text = self._extract_llm_decision(child)
                parts.append(f"  LLM: {name} tokens={child_tokens}")
                if output_text:
                    parts.append(f"    -> {output_text}")
                if child.error:
                    parts.append(f"    ERROR: {child.error[:200]}")

            elif run_type == "chain" and child.error:
                # Only log chains with errors
                parts.append(f"  CHAIN: {name} ERROR: {child.error[:200]}")

        if len(descendants) > _MAX_CHILD_RUNS:
            parts.append(
                f"  ... {len(descendants) - _MAX_CHILD_RUNS} more runs truncated"
            )

        return "\n".join(parts)

    def _extract_llm_decision(self, child) -> str:
        """Extract the key decision from an LLM run output."""
        if not child.outputs or not isinstance(child.outputs, dict):
            return ""
        generations = child.outputs.get("generations", [[]])
        if not generations or not generations[0]:
            return ""
        gen = generations[0][0] if isinstance(generations[0], list) else generations[0]
        if not isinstance(gen, dict):
            return ""
        msg = gen.get("message", {})
        if not isinstance(msg, dict):
            return ""
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            tc_names = [tc.get("name", "?") for tc in tool_calls]
            return f"called tools: {tc_names}"
        content = msg.get("content", "")
        return _truncate(content, 200)

    def _run_to_dict(self, run) -> dict[str, Any]:
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
        }



def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len chars."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _load_local_trace(trace_path: str) -> str:
    """Load a local trace JSON file as fallback diagnostic text."""
    if not trace_path:
        return ""
    path = Path(trace_path)
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text())
        parts: list[str] = []
        parts.append(f"Duration: {data.get('duration_seconds', '?')}s")
        parts.append(f"Steps: {data.get('step_count', '?')}")
        for step in data.get("steps", [])[:20]:
            step_type = step.get("type", "unknown")
            if step_type == "tool_call":
                name = step.get("name", "?")
                args = str(step.get("args_preview", ""))[:100]
                output = str(step.get("output_preview", ""))[:200]
                parts.append(f"  TOOL: {name} args={args}")
                if any(kw in output.lower() for kw in ["error", "fail", "timeout", "quota"]):
                    parts.append(f"    ERROR: {output}")
            elif step_type == "model_call":
                tool_calls = step.get("tool_calls", [])
                if tool_calls:
                    parts.append(f"  LLM: called {[tc.get('name') for tc in tool_calls]}")
        return "\n".join(parts)
    except Exception:
        return ""
