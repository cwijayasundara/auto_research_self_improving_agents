"""Two-speed evolution orchestrator.

Implements the combined approach:

INNER LOOP (fast, automatic):
  run_batch -> fetch_traces -> analyze -> reflect -> compress_memories
  -> extract_skills -> create_failure_skills -> optimize_prompt
  -> persist_state -> aggregate_metrics -> [continue/end]

OUTER LOOP (slow, coding-agent driven):
  When inner loop plateaus, the coding agent reads evolution_state/
  and makes structural changes (new tools, model swaps, architecture).

The inner loop runs autonomously until plateau detection triggers,
then signals the outer loop via evolution_state/plateau_report.md.
"""

import logging
import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from src.agent.deep_agent import create_agent, create_llm, extract_output
from src.agent.prompt_store import PromptStore
from src.agent.prompts import DEFAULT_SYSTEM_PROMPT
from src.config.settings import Settings
from src.evolution.analyzer import analyze_trajectory
from src.evolution.evolution_state_bridge import persist_evolution_state
from src.evolution.failure_skill_creator import create_failure_skills_from_batch
from src.evolution.prompt_optimizer import optimize_prompt
from src.evolution.skill_extractor import extract_skills_from_batch
from src.evolution.state import (
    AnalysisResult,
    EvolutionMetrics,
    OrchestratorState,
)
from src.memory.compression import consolidate_episodic, deduplicate_semantic
from src.memory.reflection import reflect_and_store
from src.memory.store import MemoryStore
from src.tracing.fetcher import TraceFetcher
from src.tracing.trajectory import Trajectory

logger = logging.getLogger(__name__)

# Plateau detection
MIN_IMPROVEMENT = 0.05
PLATEAU_CYCLES = 2


def _run_single_task(
    settings: Settings,
    prompt_store: PromptStore,
    memory_store: MemoryStore,
    task: str,
) -> dict[str, Any]:
    """Run the agent on a single task and return the result."""
    agent = create_agent(settings, prompt_store, memory_store, task=task)
    try:
        result = agent.invoke({"messages": [HumanMessage(content=task)]})
        output = extract_output(result)
        return {"task": task, "output": output, "status": "completed"}
    except Exception as exc:
        logger.error("Agent failed on task '%s': %s", task[:50], exc)
        return {"task": task, "output": str(exc), "status": "error"}


def build_orchestrator_graph(
    settings: Settings,
    llm: BaseChatModel,
    prompt_store: PromptStore,
    memory_store: MemoryStore,
) -> StateGraph:
    """Build the two-speed evolution orchestrator as a LangGraph StateGraph.

    Enhanced with:
    - Failure skill creation node (learns from failures, not just successes)
    - Evolution state persistence (bridges to outer-loop coding agent)
    - Skill-aware prompt optimization
    """
    trace_fetcher = TraceFetcher(settings)

    def node_run_batch(state: OrchestratorState) -> dict[str, Any]:
        """Run the agent on all tasks in the batch."""
        tasks = state["tasks"]
        cycle = state["current_cycle"]
        logger.info("")
        logger.info("=" * 60)
        logger.info("  CYCLE %d  |  Running %d tasks  |  prompt v%d",
                     cycle, len(tasks), state.get("prompt_version", 0))
        logger.info("=" * 60)

        results = []
        for i, task in enumerate(tasks, 1):
            logger.info("[%d/%d] Running: %s", i, len(tasks), task[:80])
            result = _run_single_task(settings, prompt_store, memory_store, task)
            logger.info("[%d/%d] Status: %s  (output: %d chars)",
                        i, len(tasks), result["status"], len(result["output"]))
            results.append(result)

        trajectories = []
        for r in results:
            run_id = str(uuid.uuid4())
            traj = Trajectory(
                run_id=run_id,
                task=r["task"],
                output=r["output"],
                status=r["status"],
            )
            trajectories.append(traj)

        return {"trajectories": trajectories}

    def node_fetch_traces(state: OrchestratorState) -> dict[str, Any]:
        """Fetch traces from LangSmith for the batch runs."""
        logger.info("--- Fetch LangSmith Traces ---")
        try:
            fetched = trace_fetcher.fetch_and_parse(limit=len(state["tasks"]))
            if fetched:
                logger.info("Fetched %d traces from LangSmith", len(fetched))
                return {"trajectories": fetched}
            logger.info("No traces returned, using local trajectories")
        except Exception as exc:
            logger.warning("LangSmith trace fetch failed: %s", exc)
            logger.info("Continuing with local trajectories (grading still works)")
        return {}

    def node_analyze(state: OrchestratorState) -> dict[str, Any]:
        """Analyze all trajectories through the grading pipeline."""
        logger.info("--- Analyze Trajectories ---")
        analyses: list[AnalysisResult] = []
        for traj in state["trajectories"]:
            analysis = analyze_trajectory(llm, traj)
            analyses.append(analysis)

            graders = analysis["grader_results"]
            grader_summary = "  ".join(
                f"{g['name']}={g['score']:.2f}({'PASS' if g['passed'] else 'FAIL'})"
                for g in graders
            )
            logger.info(
                "  [%s] %s -> %s (avg=%.3f)  |  %s",
                traj.run_id[:8], traj.task[:50],
                analysis["classification"].upper(),
                analysis["average_score"], grader_summary,
            )

        classifications = [a["classification"] for a in analyses]
        logger.info(
            "Analysis summary: %d successful, %d partial, %d failed",
            classifications.count("successful"),
            classifications.count("partial"),
            classifications.count("failed"),
        )
        return {"analysis_results": analyses}

    def node_reflect(state: OrchestratorState) -> dict[str, Any]:
        """Run reflection on each trajectory and store memories."""
        logger.info("--- Reflect & Store Memories ---")
        for analysis in state["analysis_results"]:
            grader_dict = {
                "average_score": analysis["average_score"],
                "classification": analysis["classification"],
                "graders": analysis["grader_results"],
            }
            reflect_and_store(
                llm=llm,
                memory_store=memory_store,
                run_id=analysis["run_id"],
                task=analysis["task"],
                output=analysis["output"],
                tool_calls=analysis["tool_calls"],
                grader_results=grader_dict,
            )
        total_ep = memory_store.count("episodic")
        total_sem = memory_store.count("semantic")
        logger.info("Memory totals: %d episodic, %d semantic", total_ep, total_sem)
        return {}

    def node_compress_memories(state: OrchestratorState) -> dict[str, Any]:
        """Compress memories: deduplicate semantic and consolidate old episodic."""
        logger.info("--- Compress Memories ---")
        deduped = deduplicate_semantic(
            memory_store,
            threshold=settings.compression_similarity_threshold,
        )
        consolidated = consolidate_episodic(
            memory_store,
            current_cycle=state["current_cycle"],
        )
        logger.info(
            "Compressed memories: %d semantic deduplicated, %d episodic consolidated",
            deduped, consolidated,
        )
        return {}

    def node_extract_skills(state: OrchestratorState) -> dict[str, Any]:
        """Extract skills from successful trajectories."""
        logger.info("--- Extract Skills (from successes) ---")
        analyses = state["analysis_results"]
        created = extract_skills_from_batch(llm, analyses, settings.skills_path)
        if created:
            for p in created:
                logger.info("  NEW SKILL: %s", p.stem)
        logger.info("Skills extracted this cycle: %d", len(created))
        return {}

    def node_create_failure_skills(state: OrchestratorState) -> dict[str, Any]:
        """Create defensive skills from failure patterns.

        This is the key new node: instead of only learning from successes,
        we also create 'antibody' skills from failures. These skills teach
        the agent what to watch for and how to avoid common pitfalls.
        """
        logger.info("--- Create Failure Skills (from failures) ---")
        analyses = state["analysis_results"]
        created = create_failure_skills_from_batch(llm, analyses, settings.skills_path)
        if created:
            for p in created:
                logger.info("  NEW DEFENSIVE SKILL: %s", p.stem)
        logger.info("Failure skills created this cycle: %d", len(created))
        return {}

    def node_optimize_prompt(state: OrchestratorState) -> dict[str, Any]:
        """Optimize the prompt based on failure analysis (skill-aware)."""
        logger.info("--- Optimize Prompt (skill-aware) ---")
        old_version = prompt_store.get_latest_version_number()
        new_version = optimize_prompt(
            llm, prompt_store, state["analysis_results"],
            skills_dir=settings.skills_path,
        )
        if new_version != old_version:
            new_prompt = prompt_store.get_current_prompt()
            logger.info(
                "Prompt upgraded: v%d -> v%d (%d chars)",
                old_version, new_version, len(new_prompt),
            )
            logger.info("  Preview: %s...", new_prompt[:150].replace("\n", " "))
        else:
            logger.info("Prompt unchanged (v%d)", old_version)
        return {"prompt_version": new_version}

    def node_persist_state(state: OrchestratorState) -> dict[str, Any]:
        """Persist evolution state for the outer-loop coding agent."""
        logger.info("--- Persist Evolution State ---")
        persist_evolution_state(
            state_dir=settings.evolution_state_path,
            skills_dir=settings.skills_path,
            cycle_metrics=state.get("cycle_metrics", []),
            analyses=state["analysis_results"],
        )
        return {}

    def node_aggregate_metrics(state: OrchestratorState) -> dict[str, Any]:
        """Aggregate metrics for the current cycle."""
        analyses = state["analysis_results"]
        scores = [a["average_score"] for a in analyses]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        from src.skills.manager import discover_skills
        all_skills = discover_skills(settings.skills_path)
        success_skills = sum(
            1 for sid in all_skills
            if not sid.startswith(("avoid-", "handle-"))
        )
        failure_skills = sum(
            1 for sid in all_skills
            if sid.startswith(("avoid-", "handle-"))
        )

        metrics = EvolutionMetrics(
            cycle=state["current_cycle"],
            avg_score=round(avg_score, 3),
            skills_learned=success_skills,
            failure_skills_created=failure_skills,
            prompt_version=state.get("prompt_version", 0),
            memories_stored=memory_store.count("episodic") + memory_store.count("semantic"),
            trajectories_analyzed=len(analyses),
        )

        cycle_metrics = [*state.get("cycle_metrics", []), metrics]
        new_cycle = state["current_cycle"] + 1

        should_continue = new_cycle < state["max_cycles"]

        # Plateau detection
        if len(cycle_metrics) >= PLATEAU_CYCLES + 1:
            recent = cycle_metrics[-PLATEAU_CYCLES:]
            improvements = [
                recent[i]["avg_score"] - recent[i - 1]["avg_score"]
                for i in range(1, len(recent))
            ]
            if all(imp < MIN_IMPROVEMENT for imp in improvements):
                logger.info("Plateau detected - stopping inner loop evolution")
                should_continue = False
                # Signal outer loop
                persist_evolution_state(
                    state_dir=settings.evolution_state_path,
                    skills_dir=settings.skills_path,
                    cycle_metrics=cycle_metrics,
                    analyses=analyses,
                    plateau_reason="Inner loop plateau: <5% improvement for 2 consecutive cycles",
                )

        # Log cycle summary
        delta_str = ""
        if len(cycle_metrics) >= 2:
            prev_score = cycle_metrics[-2]["avg_score"]
            delta = avg_score - prev_score
            direction = "+" if delta >= 0 else ""
            delta_str = f"  delta={direction}{delta:.3f}"

        logger.info("-" * 60)
        logger.info(
            "  CYCLE %d COMPLETE  |  avg_score=%.3f%s  |  skills=%d (defensive=%d)  "
            "|  prompt=v%d  |  memories=%d",
            state["current_cycle"], avg_score, delta_str,
            success_skills, failure_skills,
            state.get("prompt_version", 0),
            memory_store.count("episodic") + memory_store.count("semantic"),
        )
        if should_continue:
            logger.info("  -> Continuing to cycle %d", new_cycle)
        else:
            reason = (
                "plateau detected" if new_cycle < state["max_cycles"]
                else "max cycles reached"
            )
            logger.info("  -> Stopping (%s)", reason)
        logger.info("-" * 60)

        return {
            "cycle_metrics": cycle_metrics,
            "current_cycle": new_cycle,
            "should_continue": should_continue,
        }

    def route_continue(state: OrchestratorState) -> str:
        """Route to continue loop or end."""
        if state.get("should_continue", False):
            return "run_batch"
        return END

    # Build the graph
    graph = StateGraph(OrchestratorState)

    graph.add_node("run_batch", node_run_batch)
    graph.add_node("fetch_traces", node_fetch_traces)
    graph.add_node("analyze", node_analyze)
    graph.add_node("reflect", node_reflect)
    graph.add_node("compress_memories", node_compress_memories)
    graph.add_node("extract_skills", node_extract_skills)
    graph.add_node("create_failure_skills", node_create_failure_skills)
    graph.add_node("optimize_prompt", node_optimize_prompt)
    graph.add_node("persist_state", node_persist_state)
    graph.add_node("aggregate_metrics", node_aggregate_metrics)

    # Wire edges — enhanced pipeline with failure skills + state persistence
    graph.add_edge(START, "run_batch")
    graph.add_edge("run_batch", "fetch_traces")
    graph.add_edge("fetch_traces", "analyze")
    graph.add_edge("analyze", "reflect")
    graph.add_edge("reflect", "compress_memories")
    graph.add_edge("compress_memories", "extract_skills")
    graph.add_edge("extract_skills", "create_failure_skills")  # NEW
    graph.add_edge("create_failure_skills", "optimize_prompt")
    graph.add_edge("optimize_prompt", "persist_state")  # NEW
    graph.add_edge("persist_state", "aggregate_metrics")

    graph.add_conditional_edges(
        "aggregate_metrics",
        route_continue,
        {"run_batch": "run_batch", END: END},
    )

    return graph


def run_evolution(
    settings: Settings,
    tasks: list[str],
    max_cycles: int | None = None,
) -> list[EvolutionMetrics]:
    """Run the full inner-loop evolution.

    When the inner loop plateaus, it persists state to evolution_state/
    for the outer-loop coding agent to pick up.
    """
    max_cycles = max_cycles or settings.max_evolution_cycles
    llm = create_llm(settings)
    prompt_store = PromptStore(settings.prompts_path)
    memory_store = MemoryStore(settings.memory_path)

    if prompt_store.get_latest_version_number() == 0:
        prompt_store.add_version(DEFAULT_SYSTEM_PROMPT, score=None)

    graph = build_orchestrator_graph(settings, llm, prompt_store, memory_store)
    compiled = graph.compile()

    initial_state: OrchestratorState = {
        "tasks": tasks,
        "current_cycle": 0,
        "max_cycles": max_cycles,
        "trajectories": [],
        "analysis_results": [],
        "cycle_metrics": [],
        "prompt_version": prompt_store.get_latest_version_number(),
        "should_continue": True,
    }

    logger.info("Starting inner-loop evolution: %d tasks, max %d cycles", len(tasks), max_cycles)
    result = compiled.invoke(initial_state)

    metrics = result.get("cycle_metrics", [])
    logger.info("Inner-loop evolution complete: %d cycles executed", len(metrics))
    return metrics
