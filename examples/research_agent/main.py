"""Research agent using evoagent library — demonstrates full Karpathy loop."""

import argparse
import json

from evoagent import EvoAgentConfig
from evoagent.graders import EfficiencyGrader
from evoagent.memory import FileMemoryStore
from evoagent.skills import SkillManager


def main():
    parser = argparse.ArgumentParser(description="Self-improving research agent")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run").add_argument("task")
    evolve = sub.add_parser("evolve")
    evolve.add_argument("--tasks", required=True)
    evolve.add_argument("--cycles", type=int, default=3)
    sub.add_parser("sleep-review")

    args = parser.parse_args()
    config = EvoAgentConfig(base_dir="./data")

    if args.command == "run":
        print(f"Running task: {args.task}")
        print("Implement your AgentFactory to use the full pipeline.")
        print("See README.md for details.")

    elif args.command == "evolve":
        tasks = json.loads(open(args.tasks).read())
        print(f"Running evolution: {len(tasks)} tasks, {args.cycles} cycles")
        print("Implement your AgentFactory to use the full pipeline.")

    elif args.command == "sleep-review":
        print("Running sleep-time review...")
        print("Implement with: from evoagent.evolution import run_sleep_review")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
