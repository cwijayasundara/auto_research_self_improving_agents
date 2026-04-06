"""CLI commands for the auto-research self-improving agent.

Provides: run, evolve, prompts, skills, memory, state
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.agent.deep_agent import create_agent, create_llm, extract_output
from src.agent.prompt_store import PromptStore
from src.config.settings import Settings, configure_logging, export_langsmith_env, load_settings
from src.evolution.orchestrator import run_evolution
from evoagent.memory.store import FileMemoryStore
from evoagent.skills.manager import SkillManager

logger = logging.getLogger(__name__)


def cmd_run(settings: Settings, task: str) -> None:
    """Run the agent on a single task, then grade and learn from the result."""
    import time
    import uuid

    from src.evolution.analyzer import analyze_trajectory
    from src.memory.reflection import reflect_and_store
    from evoagent.tracing.trajectory import TrajectoryRecord

    prompt_store = PromptStore(settings.prompts_path)
    memory_store = FileMemoryStore(settings.memory_path)

    if prompt_store.get_latest_version_number() == 0:
        from src.agent.prompts import DEFAULT_SYSTEM_PROMPT

        prompt_store.add_version(DEFAULT_SYSTEM_PROMPT, score=None)

    # --- Run the agent ---
    agent = create_agent(settings, prompt_store, memory_store, task=task)
    start = time.monotonic()
    result = agent.invoke({"messages": [HumanMessage(content=task)]})
    latency = time.monotonic() - start
    output = extract_output(result)

    print("\n" + "=" * 60)
    print("AGENT OUTPUT")
    print("=" * 60)
    print(output)

    # --- Grade the result ---
    messages = result.get("messages", [])
    total_steps = len(messages)
    total_tokens = sum(len(getattr(m, "content", "") or "") // 4 for m in messages)

    traj = TrajectoryRecord(
        run_id=str(uuid.uuid4()),
        task=task,
        output=output,
        status="completed",
        total_tokens=total_tokens,
        total_steps=total_steps,
        latency_seconds=latency,
        tool_call_count=0,
    )

    llm = create_llm(settings)
    try:
        from src.tools.search import create_search_tool

        search_tool = create_search_tool(settings)
    except Exception:
        search_tool = None

    try:
        analysis = analyze_trajectory(llm, traj, search_tool=search_tool)
        graders = analysis["grader_results"]
        print("\n" + "-" * 60)
        print("GRADING")
        print("-" * 60)
        for g in graders:
            status = "PASS" if g.passed else "FAIL"
            print(f"  {g.name}: {g.score:.2f} ({status})")
        print(f"  OVERALL: {analysis['average_score']:.3f} ({analysis['classification']})")
    except Exception as exc:
        logger.error("Grading failed: %s", exc)
        return

    # --- Score the current prompt version ---
    current_version = prompt_store.get_latest_version_number()
    prompt_store.update_score(current_version, analysis["average_score"])

    # --- Log per-dimension feedback on failures ---
    failed_dims = [g for g in graders if not g.passed]
    if failed_dims:
        feedback = "; ".join(
            f"{g.name}={g.score:.2f}: {g.reasoning[:80]}" for g in failed_dims
        )
        prompt_store.append_feedback(current_version, feedback)

    # --- Capture trace path and harness config version ---
    trace_path = ""
    trace_file = settings.traces_path / f"{traj.run_id}.json"
    if trace_file.exists():
        trace_path = str(trace_file)

    from src.evolution.harness_config import HarnessConfigStore

    harness_store = HarnessConfigStore(settings.harness_config_path)
    harness_version = harness_store.get_latest_version()

    if harness_version > 0:
        harness_store.update_score(harness_version, analysis["average_score"])

    # --- Append to run log for background evolution daemon ---
    try:
        from dataclasses import asdict

        from src.evolution.run_log import RunLog, RunLogEntry

        run_log = RunLog(settings.evolution_state_path / "run_log.jsonl")
        run_log.append(
            RunLogEntry(
                run_id=traj.run_id,
                task=task,
                output=output,
                classification=analysis["classification"],
                average_score=analysis["average_score"],
                grader_results=[asdict(g) for g in graders],
                prompt_version=current_version,
                harness_config_version=harness_version,
                trace_path=trace_path,
            )
        )
    except Exception as exc:
        logger.error("Run log append failed: %s", exc)

    # --- Learn from the result ---
    try:
        from dataclasses import asdict

        grader_dict = {
            "average_score": analysis["average_score"],
            "classification": analysis["classification"],
            "graders": [asdict(g) for g in graders],
        }
        reflect_and_store(
            llm=llm,
            memory_store=memory_store,
            run_id=traj.run_id,
            task=task,
            output=output,
            tool_calls=[],
            grader_results=grader_dict,
        )
        print(f"\n  Reflection stored. Memories: {memory_store.count('episodic')} episodic, "
              f"{memory_store.count('semantic')} semantic")
    except Exception as exc:
        logger.error("Reflection failed: %s", exc)

    # Hint about daemon if not running
    try:
        from src.evolution.run_log import RunLog
        run_log_check = RunLog(settings.evolution_state_path / "run_log.jsonl")
        pending = run_log_check.count_unprocessed()
        if pending >= 3:
            print(f"\n  {pending} unprocessed runs. Start the daemon to auto-evolve:")
            print("    python -m src evolve-daemon")
    except Exception:
        pass


def cmd_evolve(settings: Settings, tasks_file: str, max_cycles: int) -> None:
    """Run the inner-loop evolution."""
    tasks_path = Path(tasks_file)
    if not tasks_path.exists():
        print(f"Tasks file not found: {tasks_path}")
        sys.exit(1)

    with open(tasks_path) as f:
        task_data = json.load(f)

    raw_tasks = task_data if isinstance(task_data, list) else task_data.get("tasks", [])
    # Normalize: support both ["task string", ...] and [{"task": "..."}, ...]
    tasks: list[str] = []
    for t in raw_tasks:
        if isinstance(t, str):
            tasks.append(t)
        elif isinstance(t, dict) and "task" in t:
            tasks.append(t["task"])
        else:
            logger.warning("Skipping unrecognized task format: %s", type(t))
    if not tasks:
        print("No tasks found in file")
        sys.exit(1)

    print(f"Starting evolution with {len(tasks)} tasks, max {max_cycles} cycles")
    print("Inner loop will persist state to evolution_state/ for the outer loop")
    metrics = run_evolution(settings, tasks, max_cycles)

    print("\n" + "=" * 60)
    print("EVOLUTION RESULTS")
    print("=" * 60)
    for m in metrics:
        print(
            f"  Cycle {m['cycle']}: score={m['avg_score']:.3f}, "
            f"skills={m['skills_learned']}, failure_skills={m.get('failure_skills_created', 0)}, "
            f"prompt=v{m['prompt_version']}, memories={m['memories_stored']}"
        )

    # Check for plateau
    if len(metrics) >= 2:
        last_delta = metrics[-1]["avg_score"] - metrics[-2]["avg_score"]
        if last_delta < 0.05:
            print("\nInner loop plateaued. Check evolution_state/plateau_report.md")
            print("The outer-loop coding agent should now make structural changes.")


def cmd_prompts(settings: Settings) -> None:
    """Show prompt version history."""
    prompt_store = PromptStore(settings.prompts_path)
    versions = prompt_store.get_all_versions()

    if not versions:
        print("No prompt versions found.")
        return

    print("Prompt Version History")
    print("=" * 60)
    for v in versions:
        score_str = f"{v.score:.3f}" if v.score is not None else "unscored"
        parent_str = f" (from v{v.parent_version})" if v.parent_version else ""
        print(f"  v{v.version}: score={score_str}{parent_str} [{v.timestamp}]")
        if v.feedback_summary:
            print(f"    feedback: {v.feedback_summary[:80]}")

    current = prompt_store.get_current_prompt()
    print(f"\nCurrent prompt ({len(current)} chars):")
    print(f"  {current[:200]}...")


def cmd_skills(settings: Settings) -> None:
    """List all learned skills (success-derived and failure-derived)."""
    skills = SkillManager(settings.skills_path).discover()
    if not skills:
        print("No skills learned yet.")
        return
    for skill_id, skill in sorted(skills.items()):
        print(f"  {skill_id}: {skill['name']} - {skill['description']}")


def cmd_memory(settings: Settings) -> None:
    """Browse stored memories."""
    memory_store = FileMemoryStore(settings.memory_path)

    episodic_count = memory_store.count("episodic")
    semantic_count = memory_store.count("semantic")

    print("Memory Store")
    print("=" * 60)
    print(f"Episodic memories: {episodic_count}")
    print(f"Semantic memories: {semantic_count}")

    if episodic_count > 0:
        print("\nRecent Episodic Memories:")
        for mem in memory_store.list_all("episodic")[-5:]:
            task = mem.get("task", "unknown")[:60]
            score = mem.get("score", "?")
            print(f"  [{mem.get('run_id', '?')[:8]}] {task} (score={score})")

    if semantic_count > 0:
        print("\nRecent Semantic Memories:")
        for mem in memory_store.list_all("semantic")[-5:]:
            content = mem.get("content", str(mem))[:80]
            mem_type = mem.get("type", "unknown")
            print(f"  [{mem_type}] {content}")


def cmd_sleep_review(settings: Settings) -> None:
    """Run offline cross-run trace analysis (sleep-time compute)."""
    from src.agent.deep_agent import create_llm
    from evoagent.evolution.sleep_review import run_sleep_review

    memory_store = FileMemoryStore(settings.memory_path)
    llm = create_llm(settings)

    print("Running sleep-time review (cross-run trace analysis)...")
    result = run_sleep_review(llm, memory=memory_store, traces_dir=settings.traces_path)

    if result["status"] == "no_traces":
        print("No traces found. Run the agent first to generate traces.")
        return

    print("\n" + "=" * 60)
    print("SLEEP REVIEW RESULTS")
    print("=" * 60)

    if result.get("meta_instructions"):
        print("\nMeta-Instructions (stored as semantic memories):")
        for i, instr in enumerate(result["meta_instructions"], 1):
            print(f"  {i}. {instr}")

    if result.get("recurring_failures"):
        print("\nRecurring Failures:")
        for f in result["recurring_failures"]:
            print(f"  - {f}")

    if result.get("consistent_successes"):
        print("\nConsistent Successes:")
        for s in result["consistent_successes"]:
            print(f"  - {s}")


def cmd_state(settings: Settings) -> None:
    """Show current evolution state and daemon activity."""
    state_dir = settings.evolution_state_path
    if not state_dir.exists():
        print("No evolution state found. Run some tasks first, then start the daemon.")
        return

    print("Evolution State")
    print("=" * 60)

    for md_file in sorted(state_dir.glob("*.md")):
        print(f"\n--- {md_file.name} ---")
        content = md_file.read_text()
        # Print first 30 lines
        lines = content.split("\n")
        for line in lines[:30]:
            print(f"  {line}")
        if len(lines) > 30:
            print(f"  ... ({len(lines) - 30} more lines)")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="auto-research-self-improving",
        description=(
            "Self-improving research agent. Every run grades, scores, and learns. "
            "The background daemon continuously evolves prompts, harness, and skills "
            "from real user interactions — no explicit 'evolve' call needed."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    run_parser = subparsers.add_parser(
        "run",
        help="Run agent on a task (grades, scores prompt+harness, reflects, logs for daemon)",
    )
    run_parser.add_argument("task", help="Task description")

    # evolve-daemon (primary evolution path)
    daemon_parser = subparsers.add_parser(
        "evolve-daemon",
        help="[PRIMARY] Background daemon — continuously evolves prompt, harness, and skills",
    )
    daemon_parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between polling the run log (default: 60)",
    )
    daemon_parser.add_argument(
        "--min-runs",
        type=int,
        default=3,
        help="Minimum unprocessed runs before triggering evolution (default: 3)",
    )

    # evolve (bootstrapping only)
    evolve_parser = subparsers.add_parser(
        "evolve",
        help="[BOOTSTRAP] Batch evolution on synthetic tasks — use for initial setup only",
    )
    evolve_parser.add_argument("--tasks-file", required=True, help="Path to tasks JSON file")
    evolve_parser.add_argument("--max-cycles", type=int, default=3, help="Maximum evolution cycles")

    # prompts
    subparsers.add_parser("prompts", help="Show prompt version history with scores")

    # skills
    subparsers.add_parser("skills", help="List all learned skills")

    # memory
    subparsers.add_parser("memory", help="Browse stored memories")

    # state
    subparsers.add_parser("state", help="Show evolution state and daemon activity")

    # sleep-review
    subparsers.add_parser(
        "sleep-review",
        help="Run offline cross-run trace analysis (sleep-time compute)",
    )

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    settings = load_settings()
    if settings is None:
        sys.exit(1)
    configure_logging(settings)
    export_langsmith_env(settings)

    if args.command == "run":
        cmd_run(settings, args.task)
    elif args.command == "evolve":
        cmd_evolve(settings, args.tasks_file, args.max_cycles)
    elif args.command == "prompts":
        cmd_prompts(settings)
    elif args.command == "skills":
        cmd_skills(settings)
    elif args.command == "memory":
        cmd_memory(settings)
    elif args.command == "state":
        cmd_state(settings)
    elif args.command == "sleep-review":
        cmd_sleep_review(settings)
    elif args.command == "evolve-daemon":
        from src.evolution.daemon import run_daemon

        run_daemon(settings, poll_interval=args.interval, min_runs=args.min_runs)
