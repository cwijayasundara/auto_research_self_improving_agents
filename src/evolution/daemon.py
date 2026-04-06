"""Background evolution daemon.

Watches the run log for new user interaction results and triggers
lightweight evolution (prompt optimization, skill extraction) when
enough signal has accumulated. Runs as a separate process alongside
the serving agent.

Usage:
    python -m src evolve-daemon
    python -m src evolve-daemon --interval 120 --min-runs 3
"""

import logging
import signal
import time
from typing import Any

from src.agent.deep_agent import create_llm
from src.agent.prompt_store import PromptStore
from src.config.settings import Settings
from src.evolution.prompt_optimizer import optimize_prompt
from src.evolution.run_log import RunLog, RunLogEntry
from src.evolution.state import AnalysisResult
from evoagent.core.types import GraderResult
from evoagent.memory.compression import deduplicate_semantic
from evoagent.memory.store import FileMemoryStore
from src.evolution.harness_config import HarnessConfigStore
from src.evolution.harness_optimizer import optimize_harness
from evoagent.skills.extractor import extract_skills_from_batch
from evoagent.skills.manager import SkillManager

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_POLL_INTERVAL = 60  # seconds between checks
DEFAULT_MIN_RUNS = 3  # minimum unprocessed runs before triggering evolution


def _entries_to_analyses(entries: list[RunLogEntry]) -> list[AnalysisResult]:
    """Convert run log entries to AnalysisResult dicts for the optimizer."""
    analyses: list[AnalysisResult] = []
    for entry in entries:
        grader_results = [
            GraderResult(
                name=g["name"],
                score=g["score"],
                passed=g["passed"],
                reasoning=g.get("reasoning", ""),
            )
            for g in entry.grader_results
        ]
        analyses.append(
            AnalysisResult(
                run_id=entry.run_id,
                task=entry.task,
                classification=entry.classification,
                average_score=entry.average_score,
                grader_results=grader_results,
                output=entry.output,
                tool_calls=[],
            )
        )
    return analyses


def _run_evolution_cycle(
    settings: Settings,
    entries: list[RunLogEntry],
) -> dict[str, Any]:
    """Run a lightweight evolution cycle on accumulated run results.

    Unlike the full `evolve` loop, this does NOT re-run tasks. It works
    entirely on the grading results already collected from user interactions.
    """
    from src.tracing.fetcher import TraceFetcher

    llm = create_llm(settings)
    prompt_store = PromptStore(settings.prompts_path)
    memory_store = FileMemoryStore(settings.memory_path)
    skill_manager = SkillManager(settings.skills_path)
    trace_fetcher = TraceFetcher(settings)

    analyses = _entries_to_analyses(entries)
    n_partial = sum(1 for a in analyses if a["classification"] == "partial")
    n_failed = sum(1 for a in analyses if a["classification"] == "failed")
    n_success = sum(1 for a in analyses if a["classification"] == "successful")

    logger.info(
        "Evolution cycle on %d runs: %d successful, %d partial, %d failed",
        len(analyses),
        n_success,
        n_partial,
        n_failed,
    )

    result: dict[str, Any] = {
        "runs_processed": len(entries),
        "prompt_changed": False,
        "skills_extracted": 0,
        "failure_skills_created": 0,
        "harness_changed": False,
    }

    # 1. Prompt optimization (if there are failures/partials)
    old_version = prompt_store.get_latest_version_number()
    try:
        new_version = optimize_prompt(
            llm,
            prompt_store,
            analyses,
            skills_dir=settings.skills_path,
            settings=settings,
            memory_store=memory_store,
            trace_fetcher=trace_fetcher,
        )
        if new_version != old_version:
            logger.info("Prompt upgraded: v%d -> v%d", old_version, new_version)
            result["prompt_changed"] = True
    except Exception as exc:
        logger.error("Prompt optimization failed: %s", exc)

    # 2. Skill extraction from successes
    try:
        created = extract_skills_from_batch(
            llm, analyses, skill_manager, mode="success"
        )
        result["skills_extracted"] = len(created)
        for p in created:
            logger.info("New skill: %s", p.stem)
    except Exception as exc:
        logger.error("Skill extraction failed: %s", exc)

    # 3. Failure skill creation
    try:
        created = extract_skills_from_batch(
            llm, analyses, skill_manager, mode="failure"
        )
        result["failure_skills_created"] = len(created)
        for p in created:
            logger.info("New defensive skill: %s", p.stem)
    except Exception as exc:
        logger.error("Failure skill creation failed: %s", exc)

    # 4. Memory compression
    try:
        deduped = deduplicate_semantic(
            memory_store,
            threshold=settings.compression_similarity_threshold,
        )
        if deduped:
            logger.info("Compressed %d duplicate memories", deduped)
    except Exception as exc:
        logger.error("Memory compression failed: %s", exc)

    # 5. Harness optimization
    try:
        harness_store = HarnessConfigStore(settings.harness_config_path)
        old_harness_version = harness_store.get_latest_version()
        new_harness_version = optimize_harness(llm, entries, harness_store, trace_fetcher=trace_fetcher)
        if new_harness_version != old_harness_version:
            logger.info(
                "Harness config upgraded: v%d -> v%d",
                old_harness_version,
                new_harness_version,
            )
            result["harness_changed"] = True
    except Exception as exc:
        logger.error("Harness optimization failed: %s", exc)

    return result


def run_daemon(
    settings: Settings,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    min_runs: int = DEFAULT_MIN_RUNS,
) -> None:
    """Run the background evolution daemon.

    Polls the run log at `poll_interval` seconds. When `min_runs` unprocessed
    entries have accumulated, triggers a lightweight evolution cycle, then
    marks those entries as processed.
    """
    run_log = RunLog(settings.evolution_state_path / "run_log.jsonl")

    logger.info(
        "Evolution daemon started (poll=%ds, min_runs=%d)",
        poll_interval,
        min_runs,
    )
    logger.info("Watching: %s", run_log.log_path)

    # Graceful shutdown on SIGINT/SIGTERM
    shutdown = False

    def _handle_signal(signum, frame):
        nonlocal shutdown
        logger.info("Shutdown signal received, stopping after current cycle...")
        shutdown = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cycle_count = 0
    while not shutdown:
        try:
            n_pending = run_log.count_unprocessed()

            if n_pending >= min_runs:
                entries = run_log.read_unprocessed()
                cycle_count += 1
                logger.info(
                    "--- Daemon Cycle %d: processing %d runs ---",
                    cycle_count,
                    len(entries),
                )

                result = _run_evolution_cycle(settings, entries)

                # Mark as processed
                run_ids = {e.run_id for e in entries}
                run_log.mark_processed(run_ids)

                logger.info(
                    "Daemon Cycle %d complete: prompt_changed=%s, "
                    "harness_changed=%s, skills=%d, failure_skills=%d",
                    cycle_count,
                    result["prompt_changed"],
                    result.get("harness_changed", False),
                    result["skills_extracted"],
                    result["failure_skills_created"],
                )
            else:
                if n_pending > 0:
                    logger.debug(
                        "%d/%d runs pending, waiting for more...",
                        n_pending,
                        min_runs,
                    )
        except Exception as exc:
            logger.error("Daemon cycle failed: %s", exc, exc_info=True)

        # Sleep in short intervals so we can respond to shutdown signals
        for _ in range(poll_interval):
            if shutdown:
                break
            time.sleep(1)

    logger.info("Evolution daemon stopped after %d cycles", cycle_count)
