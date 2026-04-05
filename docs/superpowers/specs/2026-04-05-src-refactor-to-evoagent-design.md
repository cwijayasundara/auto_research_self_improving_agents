# Refactor src/ to Use evoagent Library

**Date:** 2026-04-05
**Status:** Approved

## Problem

The `src/` codebase and `evoagent/` library contain duplicated code. The library was extracted from `src/` but `src/` still uses its own internal copies. This refactoring makes `evoagent` the single source of truth by rewriting all `src/` modules to import from `evoagent/`.

## Approach

Full rewrite — no compatibility shims. Delete `src/` modules that are fully replaced by evoagent equivalents. Rewrite modules that need API migration (TypedDict→dataclass, free functions→class methods, etc.).

## Files Deleted

| File | Replaced By |
|---|---|
| `src/memory/store.py` | `evoagent.memory.store.FileMemoryStore` |
| `src/memory/compression.py` | `evoagent.memory.compression` |
| `src/skills/manager.py` | `evoagent.skills.manager.SkillManager` |
| `src/evolution/skill_extractor.py` | `evoagent.skills.extractor` |
| `src/evolution/failure_skill_creator.py` | `evoagent.skills.extractor` (mode="failure") |
| `src/agent/middleware.py` | `evoagent.harness.middleware` |
| `src/agent/sleep_review.py` | `evoagent.evolution.sleep_review` |
| `src/tracing/trajectory.py` | `evoagent.tracing.trajectory` |
| `src/evolution/graders/efficiency.py` | `evoagent.graders.efficiency.EfficiencyGrader` |
| `src/evolution/graders/multi_judge.py` | `evoagent.graders.multi_judge.MultiJudgeGrader` |

## Files Rewritten

| File | Key Changes |
|---|---|
| `src/evolution/state.py` | Remove types that moved to evoagent.core.types; keep LangGraph-specific state TypedDicts (AnalyzerState, OrchestratorState) but reference evoagent types |
| `src/evolution/orchestrator.py` | Use FileMemoryStore, SkillManager, dataclass GraderResult (attribute access), evoagent imports |
| `src/evolution/analyzer.py` | Use evoagent Grader protocol, dataclass GraderResult |
| `src/evolution/prompt_optimizer.py` | Delegate to evoagent.evolution.prompt_optimizer or import its components |
| `src/evolution/evolution_state_bridge.py` | Use evoagent.evolution.state.persist_evolution_state |
| `src/agent/deep_agent.py` | Import middleware from evoagent.harness |
| `src/memory/reflection.py` | Use FileMemoryStore |
| `src/cli/commands.py` | Use evoagent imports for memory, skills, sleep review |
| `src/tracing/fetcher.py` | Use TrajectoryRecord instead of Trajectory |
| `src/evolution/graders/claim_verification.py` | Use dataclass GraderResult (attribute access) |
| `src/evolution/graders/quality.py` | Use dataclass GraderResult |
| `src/evolution/graders/task_completion.py` | Use dataclass GraderResult |

## Key API Migrations

| Old (src/) | New (evoagent/) |
|---|---|
| `r["score"]`, `r["passed"]` | `r.score`, `r.passed` |
| `MemoryStore(path)` | `FileMemoryStore(path)` |
| `store.list_all("ns")` returns `list[dict]` | returns `dict[str, dict]` |
| `discover_skills(path)` | `SkillManager(path).discover()` |
| `create_skill(path, name, desc, content)` | `SkillManager(path).create(name, desc, content)` |
| `grade_efficiency(metrics)` | `EfficiencyGrader().grade(task, output, metrics=metrics)` |
| `multi_judge_task_completion(llm, task, out)` | `MultiJudgeGrader(llm, "task_completion", judge_prompts=[...]).grade(task, out)` |
| `Trajectory` | `TrajectoryRecord` |
| `TrajectoryMetrics` (from tracing) | `TrajectoryMetrics` (from evoagent.core.types) |
| `compress_memory_context(...)` | `compress_context(...)` |
| `GraderResult` TypedDict | `GraderResult` dataclass |

## Files Unchanged

- `src/agent/prompts.py` — research-specific prompts
- `src/agent/prompt_store.py` — already implements PromptStore pattern
- `src/agent/subagents.py` — research-specific
- `src/tools/search.py` — research-specific
- `src/config/settings.py` — app-level config
