"""Evolution state bridge.

Persists inner-loop evolution state to disk so the outer-loop coding agent
(Claude Code, Cursor, Codex) can read it and make informed structural decisions.

This bridges the gap between Karpathy's autoresearch approach (memoryless
hill-climbing) and the self-evolving agents' memory system by exposing
the agent's learned knowledge to the coding agent.

The coding agent reads evolution_state/ before proposing code changes,
turning blind experimentation into informed search.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.evolution.state import AnalysisResult, EvolutionMetrics
from evoagent.skills.manager import SkillManager

logger = logging.getLogger(__name__)


def _write_json(path: Path, data: Any) -> None:
    """Write JSON to file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_markdown(path: Path, content: str) -> None:
    """Write markdown to file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def persist_failure_analysis(
    state_dir: Path,
    analyses: list[AnalysisResult],
) -> None:
    """Write deduplicated failure patterns to evolution_state/failures.md.

    The coding agent reads this to understand what the inner loop couldn't fix.
    """
    failed = [a for a in analyses if a["classification"] in ("failed", "partial")]
    if not failed:
        content = (
            "# Failure Analysis\n\n"
            "No failures detected in the most recent evolution cycle.\n"
            "All trajectories classified as successful.\n"
        )
        _write_markdown(state_dir / "failures.md", content)
        return

    lines = [
        "# Failure Analysis",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Failed/partial trajectories: {len(failed)}",
        "",
        "## Failure Patterns",
        "",
    ]

    # Collect and deduplicate issues
    issues: list[dict[str, str]] = []
    seen: set[str] = set()
    for analysis in failed:
        for grader in analysis["grader_results"]:
            if not grader.passed:
                key = f"{grader.name}:{grader.reasoning[:50]}"
                if key not in seen:
                    seen.add(key)
                    issues.append(
                        {
                            "grader": grader.name,
                            "score": f"{grader.score:.3f}",
                            "reasoning": grader.reasoning,
                            "task": analysis["task"][:100],
                        }
                    )

    for issue in issues[:15]:
        lines.append(f"### [{issue['grader']}] score={issue['score']}")
        lines.append(f"- **Task**: {issue['task']}")
        lines.append(f"- **Issue**: {issue['reasoning']}")
        lines.append("")

    lines.extend(
        [
            "## Recommendations for Coding Agent",
            "",
            "Based on the failure patterns above, consider:",
            "",
        ]
    )

    # Generate recommendations based on which graders failed most
    grader_failures: dict[str, int] = {}
    for analysis in failed:
        for grader in analysis["grader_results"]:
            if not grader.passed:
                grader_failures[grader.name] = grader_failures.get(grader.name, 0) + 1

    for grader_name, count in sorted(grader_failures.items(), key=lambda x: -x[1]):
        if grader_name == "task_completion":
            lines.append(
                f"- **Task completion failed {count}x**: Consider adding new tools, "
                "improving search strategies, or restructuring the agent architecture."
            )
        elif grader_name == "efficiency":
            lines.append(
                f"- **Efficiency failed {count}x**: Consider optimizing token usage, "
                "reducing unnecessary tool calls, or implementing caching."
            )
        elif grader_name == "quality":
            lines.append(
                f"- **Quality failed {count}x**: Consider improving output formatting, "
                "adding citation requirements, or enhancing synthesis prompts."
            )

    _write_markdown(state_dir / "failures.md", "\n".join(lines))
    logger.info("Persisted %d failure patterns to evolution_state/failures.md", len(issues))


def persist_hypotheses(
    state_dir: Path,
    cycle_metrics: list[EvolutionMetrics],
    analyses: list[AnalysisResult],
) -> None:
    """Write experiment history and hypotheses to evolution_state/hypotheses.md.

    Tracks what's been tried and the outcomes, giving the coding agent
    context to avoid repeating failed experiments.
    """
    lines = [
        "# Experiment Hypotheses & History",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Total cycles completed: {len(cycle_metrics)}",
        "",
        "## Cycle History",
        "",
    ]

    for m in cycle_metrics:
        lines.append(
            f"- **Cycle {m['cycle']}**: score={m['avg_score']:.3f}, "
            f"skills={m['skills_learned']}, failure_skills={m.get('failure_skills_created', 0)}, "
            f"prompt=v{m['prompt_version']}, memories={m['memories_stored']}"
        )

    # Score progression
    if len(cycle_metrics) >= 2:
        lines.extend(["", "## Score Progression", ""])
        for i in range(1, len(cycle_metrics)):
            prev = cycle_metrics[i - 1]["avg_score"]
            curr = cycle_metrics[i]["avg_score"]
            delta = curr - prev
            direction = "+" if delta >= 0 else ""
            lines.append(f"- Cycle {i - 1} -> {i}: {direction}{delta:.3f}")

    # Current state assessment
    lines.extend(["", "## Current Assessment", ""])
    if cycle_metrics:
        latest = cycle_metrics[-1]
        if latest["avg_score"] >= 0.9:
            lines.append("Agent is performing well. Focus on edge cases and efficiency.")
        elif latest["avg_score"] >= 0.7:
            lines.append("Agent shows promise but has room for improvement.")
        else:
            lines.append("Agent needs significant improvement. Consider structural changes.")

    _write_markdown(state_dir / "hypotheses.md", "\n".join(lines))
    logger.info("Persisted hypotheses to evolution_state/hypotheses.md")


def persist_skills_summary(
    state_dir: Path,
    skills_dir: Path,
) -> None:
    """Write a summary of all learned skills to evolution_state/skills_summary.md."""
    skills = SkillManager(skills_dir).discover()

    lines = [
        "# Learned Skills Summary",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Total skills: {len(skills)}",
        "",
    ]

    if not skills:
        lines.append("No skills learned yet.")
    else:
        # Separate success-derived and failure-derived skills
        success_skills = []
        failure_skills = []
        for skill_id, skill in sorted(skills.items()):
            if skill_id.startswith(("avoid-", "handle-")):
                failure_skills.append((skill_id, skill))
            else:
                success_skills.append((skill_id, skill))

        if success_skills:
            lines.extend(["## Skills from Successes", ""])
            for skill_id, skill in success_skills:
                lines.append(f"- **{skill['name']}** (`{skill_id}`): {skill['description']}")
            lines.append("")

        if failure_skills:
            lines.extend(["## Defensive Skills from Failures", ""])
            for skill_id, skill in failure_skills:
                lines.append(f"- **{skill['name']}** (`{skill_id}`): {skill['description']}")
            lines.append("")

    _write_markdown(state_dir / "skills_summary.md", "\n".join(lines))
    logger.info("Persisted skills summary to evolution_state/skills_summary.md")


def persist_plateau_report(
    state_dir: Path,
    cycle_metrics: list[EvolutionMetrics],
    reason: str,
) -> None:
    """Write a plateau report signaling the outer loop to take over.

    This is the handoff point: the inner loop has exhausted its
    prompt/skill/memory levers and needs structural code changes.
    """
    lines = [
        "# Inner Loop Plateau Report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        f"Reason: {reason}",
        "",
        "## Final Metrics",
        "",
    ]

    if cycle_metrics:
        latest = cycle_metrics[-1]
        lines.append(f"- Final score: {latest['avg_score']:.3f}")
        lines.append(f"- Skills learned: {latest['skills_learned']}")
        lines.append(f"- Failure skills: {latest.get('failure_skills_created', 0)}")
        lines.append(f"- Prompt version: v{latest['prompt_version']}")
        lines.append(f"- Memories stored: {latest['memories_stored']}")
        lines.append(f"- Cycles completed: {len(cycle_metrics)}")

    lines.extend(
        [
            "",
            "## What the Inner Loop Tried",
            "",
            "The inner loop exhausted these optimization levers:",
            "- Prompt optimization (metaprompt-driven rewriting)",
            "- Skill extraction from successful runs",
            "- Defensive skill creation from failures",
            "- Memory accumulation (episodic + semantic)",
            "- Memory compression and deduplication",
            "",
            "## What the Outer Loop Should Try",
            "",
            "The coding agent should now consider structural changes:",
            "- Adding new tools (capabilities the agent lacks)",
            "- Changing the model or provider",
            "- Modifying agent architecture (routing, multi-agent)",
            "- Framework changes",
            "- Expanding the evaluation dataset",
            "",
            "Read `failures.md` and `hypotheses.md` for detailed context.",
        ]
    )

    _write_markdown(state_dir / "plateau_report.md", "\n".join(lines))
    logger.info("Persisted plateau report — outer loop should take over")


def persist_evolution_state(
    state_dir: Path,
    skills_dir: Path,
    cycle_metrics: list[EvolutionMetrics],
    analyses: list[AnalysisResult],
    plateau_reason: str | None = None,
) -> None:
    """Persist all evolution state for the outer-loop coding agent.

    This is called at the end of each inner-loop cycle to keep the
    evolution_state/ directory up to date.
    """
    state_dir.mkdir(parents=True, exist_ok=True)

    persist_failure_analysis(state_dir, analyses)
    persist_hypotheses(state_dir, cycle_metrics, analyses)
    persist_skills_summary(state_dir, skills_dir)

    if plateau_reason:
        persist_plateau_report(state_dir, cycle_metrics, plateau_reason)

    logger.info("Evolution state persisted to %s", state_dir)
