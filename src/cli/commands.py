"""CLI commands for the auto-research self-improving agent.

Provides: run, evolve, prompts, skills, memory, state
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.agent.deep_agent import create_agent, extract_output
from src.agent.prompt_store import PromptStore
from src.config.settings import Settings, configure_logging, export_langsmith_env, load_settings
from src.evolution.orchestrator import run_evolution
from src.memory.store import MemoryStore
from src.skills.manager import list_skills

logger = logging.getLogger(__name__)


def cmd_run(settings: Settings, task: str) -> None:
    """Run the agent on a single task."""
    prompt_store = PromptStore(settings.prompts_path)
    memory_store = MemoryStore(settings.memory_path)

    if prompt_store.get_latest_version_number() == 0:
        from src.agent.prompts import DEFAULT_SYSTEM_PROMPT
        prompt_store.add_version(DEFAULT_SYSTEM_PROMPT, score=None)

    agent = create_agent(settings, prompt_store, memory_store, task=task)
    result = agent.invoke({"messages": [HumanMessage(content=task)]})
    output = extract_output(result)

    print("\n" + "=" * 60)
    print("AGENT OUTPUT")
    print("=" * 60)
    print(output)


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
    print(list_skills(settings.skills_path))


def cmd_memory(settings: Settings) -> None:
    """Browse stored memories."""
    memory_store = MemoryStore(settings.memory_path)

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


def cmd_state(settings: Settings) -> None:
    """Show current evolution state (for outer-loop debugging)."""
    state_dir = settings.evolution_state_path
    if not state_dir.exists():
        print("No evolution state found. Run 'evolve' first.")
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
            "Self-improving research agent with two-speed evolution: "
            "inner loop (prompt/skill/memory) + outer loop (code changes)"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run
    run_parser = subparsers.add_parser("run", help="Run agent on a single task")
    run_parser.add_argument("task", help="Task description")

    # evolve
    evolve_parser = subparsers.add_parser("evolve", help="Run inner-loop evolution")
    evolve_parser.add_argument(
        "--tasks-file", required=True, help="Path to tasks JSON file"
    )
    evolve_parser.add_argument(
        "--max-cycles", type=int, default=3, help="Maximum evolution cycles"
    )

    # prompts
    subparsers.add_parser("prompts", help="Show prompt version history")

    # skills
    subparsers.add_parser("skills", help="List all learned skills")

    # memory
    subparsers.add_parser("memory", help="Browse stored memories")

    # state
    subparsers.add_parser("state", help="Show evolution state for outer loop")

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
