# Refactor src/ to Use evoagent Library — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all duplicated code in `src/` with imports from the `evoagent` library, making evoagent the single source of truth.

**Architecture:** Bottom-up refactoring — start with leaf modules (graders, types), then mid-level (memory, skills, tracing), then top-level (analyzer, orchestrator, CLI). Delete fully-replaced files at each step.

**Tech Stack:** Python 3.12+, evoagent library (already in repo at `evoagent/`)

**Spec:** `docs/superpowers/specs/2026-04-05-src-refactor-to-evoagent-design.md`

---

## Critical Context

The main API migration is `GraderResult` changing from a `TypedDict` (dict access `r["score"]`) to a `dataclass` (attribute access `r.score`). This affects ~15 files. The `MemoryStore.list_all()` return type also changed from `list[dict]` to `dict[str, dict]`.

All work happens in `/Users/chamindawijayasundara/Documents/rnd_2026/auto_research_self_improving_agents`.

Run tests with: `PYTHONPATH=. python -m pytest tests/ -x -q`

---

## Task 1: Rewrite src/evolution/state.py — Remove migrated types, keep LangGraph state

**Files:**
- Modify: `src/evolution/state.py`

The old `state.py` defines both shared types (`GraderResult`, `EvolutionMetrics`) and LangGraph-specific state schemas (`AnalyzerState`, `OrchestratorState`). The shared types now live in `evoagent.core.types`. Keep only LangGraph state schemas, importing types from evoagent.

- [ ] **Step 1: Rewrite `src/evolution/state.py`**

```python
"""LangGraph state schemas for the evolution pipeline.

Shared types (GraderResult, etc.) now live in evoagent.core.types.
This file only contains LangGraph-specific state TypedDicts.
"""

from typing import Any, TypedDict

from evoagent.core.types import GraderResult
from evoagent.tracing.trajectory import TrajectoryRecord


class AnalysisResult(TypedDict):
    """Aggregated analysis of a single trajectory."""

    run_id: str
    task: str
    classification: str  # "successful", "partial", "failed"
    average_score: float
    grader_results: list[GraderResult]
    output: str
    tool_calls: list[dict[str, Any]]


class AnalyzerState(TypedDict):
    """State for the trajectory analyzer LangGraph workflow."""

    trajectory: TrajectoryRecord
    task_completion: GraderResult
    efficiency: GraderResult
    quality: GraderResult
    claim_verification: GraderResult
    classification: str
    average_score: float


class EvolutionMetrics(TypedDict):
    """Metrics for a single evolution cycle."""

    cycle: int
    avg_score: float
    skills_learned: int
    failure_skills_created: int
    prompt_version: int
    memories_stored: int
    trajectories_analyzed: int


class OrchestratorState(TypedDict):
    """State for the top-level evolution orchestrator."""

    tasks: list[str]
    current_cycle: int
    max_cycles: int
    trajectories: list[TrajectoryRecord]
    analysis_results: list[AnalysisResult]
    cycle_metrics: list[EvolutionMetrics]
    prompt_version: int
    should_continue: bool


class OrchestratorInput(TypedDict):
    """Public input to the orchestrator."""

    tasks: list[str]
    max_cycles: int


class OrchestratorOutput(TypedDict):
    """Public output from the orchestrator."""

    cycle_metrics: list[EvolutionMetrics]
    analysis_results: list[AnalysisResult]
```

**Important:** `GraderResult` is now a dataclass, not a TypedDict. The `AnalysisResult` TypedDict references `list[GraderResult]` — this is fine since TypedDict annotations are just hints, and LangGraph state reducers handle it.

- [ ] **Step 2: Verify imports work**

```bash
PYTHONPATH=. python -c "from src.evolution.state import AnalysisResult, AnalyzerState, OrchestratorState, EvolutionMetrics; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/evolution/state.py
git commit -m "refactor: rewrite src/evolution/state.py to import types from evoagent

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Rewrite leaf graders — quality.py, task_completion.py, claim_verification.py, fact_checker.py

**Files:**
- Modify: `src/evolution/graders/quality.py`
- Modify: `src/evolution/graders/task_completion.py`
- Modify: `src/evolution/graders/claim_verification.py`
- Modify: `src/evolution/graders/fact_checker.py`
- Delete: `src/evolution/graders/efficiency.py`
- Delete: `src/evolution/graders/multi_judge.py`

These graders import `GraderResult` from `src.evolution.state` and use `_parse_json_response` locally. Change to import from evoagent. Delete efficiency.py and multi_judge.py (fully replaced by evoagent).

- [ ] **Step 1: Rewrite `src/evolution/graders/quality.py`**

```python
"""LLM-as-judge grader: output quality."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evoagent.core.parsing import parse_llm_json
from evoagent.core.types import GraderResult
from src.agent.prompts import QUALITY_PROMPT

logger = logging.getLogger(__name__)


def grade_quality(llm: BaseChatModel, task: str, output: str) -> GraderResult:
    """Grade the quality of the agent's output."""
    prompt = QUALITY_PROMPT.format(task=task, output=output[:4000])
    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = parse_llm_json(response.content)

    score = float(parsed.get("overall_score", 0.5))
    return GraderResult(
        name="quality",
        score=round(score, 3),
        passed=score >= 0.75,
        reasoning=parsed.get("reasoning", ""),
    )
```

- [ ] **Step 2: Rewrite `src/evolution/graders/task_completion.py`**

```python
"""LLM-as-judge grader: task completion."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evoagent.core.parsing import parse_llm_json
from evoagent.core.types import GraderResult
from src.agent.prompts import TASK_COMPLETION_PROMPT

logger = logging.getLogger(__name__)


def grade_task_completion(llm: BaseChatModel, task: str, output: str) -> GraderResult:
    """Grade whether the agent completed the task."""
    prompt = TASK_COMPLETION_PROMPT.format(task=task, output=output[:4000])
    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = parse_llm_json(response.content)

    score = float(parsed.get("score", 0.5))
    return GraderResult(
        name="task_completion",
        score=score,
        passed=parsed.get("passed", score >= 0.75),
        reasoning=parsed.get("reasoning", ""),
    )
```

- [ ] **Step 3: Rewrite `src/evolution/graders/claim_verification.py`**

```python
"""Claim-level verification grader."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evoagent.core.parsing import parse_llm_json
from evoagent.core.types import GraderResult
from src.agent.prompts import CLAIM_EXTRACTION_PROMPT, CLAIM_VERIFICATION_PROMPT

logger = logging.getLogger(__name__)


def _extract_claims(llm: BaseChatModel, output: str) -> list[str]:
    try:
        prompt = CLAIM_EXTRACTION_PROMPT.replace("{output}", output[:4000])
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = parse_llm_json(response.content)
        return parsed.get("claims", [])
    except Exception:
        logger.warning("Failed to extract claims", exc_info=True)
        return []


def _verify_claims(llm: BaseChatModel, claims: list[str], output: str) -> list[dict]:
    try:
        claims_text = "\n".join(f"- {c}" for c in claims)
        prompt = CLAIM_VERIFICATION_PROMPT.replace("{claims}", claims_text).replace(
            "{output}", output[:4000]
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = parse_llm_json(response.content)
        return parsed.get("verdicts", [])
    except Exception:
        logger.warning("Failed to verify claims", exc_info=True)
        return []


def _compute_consistency_score(verdicts: list[dict]) -> float:
    if not verdicts:
        return 0.5
    supported = sum(1 for v in verdicts if v.get("verdict") == "supported")
    return supported / len(verdicts)


def grade_claims(
    llm: BaseChatModel,
    task: str,
    output: str,
    spot_check_score: float | None = None,
) -> GraderResult:
    """Grade agent output by extracting and verifying factual claims."""
    claims = _extract_claims(llm, output)
    if not claims:
        return GraderResult(
            name="claim_verification", score=0.5, passed=False,
            reasoning="No claims could be extracted from the output.",
        )

    verdicts = _verify_claims(llm, claims, output)
    if not verdicts:
        return GraderResult(
            name="claim_verification", score=0.5, passed=False,
            reasoning="Claim verification failed to produce verdicts.",
        )

    consistency = _compute_consistency_score(verdicts)
    final = 0.6 * consistency + 0.4 * spot_check_score if spot_check_score is not None else consistency

    supported_count = sum(1 for v in verdicts if v.get("verdict") == "supported")
    return GraderResult(
        name="claim_verification",
        score=final,
        passed=final >= 0.6,
        reasoning=(
            f"Consistency: {consistency:.2f} ({supported_count}/{len(verdicts)} supported). "
            + (f"Spot-check: {spot_check_score:.2f}. " if spot_check_score is not None else "")
            + f"Final: {final:.2f}."
        ),
    )
```

- [ ] **Step 4: Rewrite `src/evolution/graders/fact_checker.py`**

Replace local `_parse_json_response` with `parse_llm_json` from evoagent:

```python
"""Factual spot-check via web search."""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from evoagent.core.parsing import parse_llm_json
from src.agent.prompts import CLAIM_SELECTION_PROMPT, FACT_CHECK_PROMPT

logger = logging.getLogger(__name__)

VERDICT_SCORES = {"corroborated": 1.0, "inconclusive": 0.5, "contradicted": 0.0}


def _select_verifiable_claims(llm: BaseChatModel, claims: list[str]) -> list[str]:
    try:
        claims_text = "\n".join(f"- {c}" for c in claims)
        prompt = CLAIM_SELECTION_PROMPT.replace("{claims}", claims_text)
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = parse_llm_json(response.content)
        selected = parsed.get("selected", [])
        return selected if selected else claims[:2]
    except Exception:
        logger.warning("Failed to select verifiable claims", exc_info=True)
        return claims[:2]


def _check_single_claim(llm: BaseChatModel, search_tool: BaseTool, claim: str) -> str | None:
    try:
        search_results = search_tool._run(claim)
    except Exception:
        logger.warning("Search failed for claim: %s", claim, exc_info=True)
        return None

    if not search_results:
        return None

    search_text = str(search_results) if not isinstance(search_results, str) else search_results

    try:
        prompt = FACT_CHECK_PROMPT.replace("{claim}", claim).replace(
            "{search_results}", search_text[:3000]
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed = parse_llm_json(response.content)
        verdict = parsed.get("verdict", "")
        return verdict if verdict in VERDICT_SCORES else "inconclusive"
    except Exception:
        logger.warning("LLM fact-check failed for claim: %s", claim, exc_info=True)
        return None


def _compute_spot_check_score(verdicts: list[str]) -> float | None:
    if not verdicts:
        return None
    return sum(VERDICT_SCORES.get(v, 0.5) for v in verdicts) / len(verdicts)


def spot_check_claims(llm: BaseChatModel, search_tool: BaseTool, claims: list[str]) -> float | None:
    """Spot-check claims by searching the web."""
    if not claims:
        return None
    selected = _select_verifiable_claims(llm, claims)
    verdicts: list[str] = []
    for claim in selected:
        verdict = _check_single_claim(llm, search_tool, claim)
        if verdict is not None:
            verdicts.append(verdict)
    return _compute_spot_check_score(verdicts)
```

- [ ] **Step 5: Delete fully-replaced graders**

```bash
rm src/evolution/graders/efficiency.py src/evolution/graders/multi_judge.py
```

- [ ] **Step 6: Verify grader imports**

```bash
PYTHONPATH=. python -c "
from src.evolution.graders.quality import grade_quality
from src.evolution.graders.task_completion import grade_task_completion
from src.evolution.graders.claim_verification import grade_claims, _extract_claims
from src.evolution.graders.fact_checker import spot_check_claims
print('All graders OK')
"
```

- [ ] **Step 7: Commit**

```bash
git add src/evolution/graders/
git commit -m "refactor: rewrite graders to use evoagent types and parsing, delete replaced graders

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Rewrite src/evolution/analyzer.py — Use evoagent GraderResult (dataclass)

**Files:**
- Modify: `src/evolution/analyzer.py`

The analyzer uses GraderResult as TypedDict (`g["score"]`). Rewrite to use dataclass attribute access (`g.score`). Keep the LangGraph structure since it uses research-specific graders.

- [ ] **Step 1: Rewrite `src/evolution/analyzer.py`**

```python
"""Trajectory analyzer LangGraph workflow.

Runs four graders in parallel and classifies trajectories.
Uses evoagent types (dataclass GraderResult).
"""

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from evoagent.core.types import GraderResult
from src.evolution.graders.claim_verification import _extract_claims, grade_claims
from src.evolution.graders.fact_checker import spot_check_claims
from src.evolution.graders.quality import grade_quality
from src.evolution.graders.task_completion import grade_task_completion
from src.evolution.state import AnalysisResult, AnalyzerState
from evoagent.graders.efficiency import EfficiencyGrader
from evoagent.graders.multi_judge import MultiJudgeGrader
from evoagent.tracing.trajectory import TrajectoryRecord
from src.agent.prompts import (
    Q_DEPTH_PROMPT, Q_RELEVANCE_PROMPT, Q_STRUCTURE_PROMPT, QUALITY_PROMPT,
    TASK_COMPLETION_PROMPT,
    TC_ACCURACY_PROMPT, TC_COMPLETENESS_PROMPT, TC_EVIDENCE_PROMPT,
)

logger = logging.getLogger(__name__)

SUCCESSFUL_THRESHOLD = 0.75
PARTIAL_THRESHOLD = 0.5
MIN_PASS_COUNT = 2
MIN_AVERAGE_SCORE = 0.6


def classify_trajectory(grader_results: list[GraderResult]) -> tuple[str, float]:
    """Classify a trajectory based on grader results."""
    if not grader_results:
        return "failed", 0.0

    scores = [g.score for g in grader_results]
    avg_score = sum(scores) / len(scores)
    pass_count = sum(1 for g in grader_results if g.passed)

    if pass_count >= MIN_PASS_COUNT and avg_score >= MIN_AVERAGE_SCORE:
        if avg_score >= SUCCESSFUL_THRESHOLD:
            return "successful", round(avg_score, 3)
        return "partial", round(avg_score, 3)

    if avg_score >= PARTIAL_THRESHOLD:
        return "partial", round(avg_score, 3)

    return "failed", round(avg_score, 3)


def build_analyzer_graph(llm: BaseChatModel, search_tool: BaseTool | None = None) -> StateGraph:
    """Build the trajectory analyzer as a LangGraph StateGraph."""

    tc_grader = MultiJudgeGrader(
        llm, name="task_completion",
        judge_prompts=[TC_COMPLETENESS_PROMPT, TC_EVIDENCE_PROMPT, TC_ACCURACY_PROMPT],
        fallback_prompt=TASK_COMPLETION_PROMPT,
    )
    quality_grader = MultiJudgeGrader(
        llm, name="quality",
        judge_prompts=[Q_STRUCTURE_PROMPT, Q_DEPTH_PROMPT, Q_RELEVANCE_PROMPT],
        fallback_prompt=QUALITY_PROMPT,
    )
    efficiency_grader = EfficiencyGrader()

    def node_grade_task_completion(state: AnalyzerState) -> dict[str, Any]:
        traj = state["trajectory"]
        result = tc_grader.grade(traj.task, traj.output)
        return {"task_completion": result}

    def node_grade_efficiency(state: AnalyzerState) -> dict[str, Any]:
        traj = state["trajectory"]
        result = efficiency_grader.grade(
            traj.task, traj.output,
            metrics=evoagent_metrics_from_record(traj),
        )
        return {"efficiency": result}

    def node_grade_quality(state: AnalyzerState) -> dict[str, Any]:
        traj = state["trajectory"]
        result = quality_grader.grade(traj.task, traj.output)
        return {"quality": result}

    def node_grade_claims(state: AnalyzerState) -> dict[str, Any]:
        traj = state["trajectory"]
        claims = _extract_claims(llm, traj.output)
        spot_score = None
        if search_tool and claims:
            spot_score = spot_check_claims(llm, search_tool, claims)
        result = grade_claims(llm, traj.task, traj.output, spot_check_score=spot_score)
        return {"claim_verification": result}

    def node_classify(state: AnalyzerState) -> dict[str, Any]:
        grader_results = [
            state["task_completion"],
            state["efficiency"],
            state["quality"],
            state["claim_verification"],
        ]
        classification, avg_score = classify_trajectory(grader_results)
        return {"classification": classification, "average_score": avg_score}

    graph = StateGraph(AnalyzerState)
    graph.add_node("grade_task_completion", node_grade_task_completion)
    graph.add_node("grade_efficiency", node_grade_efficiency)
    graph.add_node("grade_quality", node_grade_quality)
    graph.add_node("grade_claims", node_grade_claims)
    graph.add_node("classify", node_classify)

    graph.add_edge(START, "grade_task_completion")
    graph.add_edge(START, "grade_efficiency")
    graph.add_edge(START, "grade_quality")
    graph.add_edge(START, "grade_claims")

    graph.add_edge("grade_task_completion", "classify")
    graph.add_edge("grade_efficiency", "classify")
    graph.add_edge("grade_quality", "classify")
    graph.add_edge("grade_claims", "classify")

    graph.add_edge("classify", END)

    return graph


def evoagent_metrics_from_record(traj: TrajectoryRecord):
    """Convert TrajectoryRecord fields to TrajectoryMetrics for EfficiencyGrader."""
    from evoagent.core.types import TrajectoryMetrics
    return TrajectoryMetrics(
        total_tokens=traj.total_tokens,
        total_steps=traj.total_steps,
        latency_seconds=traj.latency_seconds,
        tool_call_count=traj.tool_call_count,
    )


def analyze_trajectory(
    llm: BaseChatModel,
    trajectory: TrajectoryRecord,
    search_tool: BaseTool | None = None,
) -> AnalysisResult:
    """Analyze a single trajectory through the grading pipeline."""
    graph = build_analyzer_graph(llm, search_tool=search_tool)
    compiled = graph.compile()

    initial_state: AnalyzerState = {
        "trajectory": trajectory,
        "task_completion": GraderResult(name="", score=0, passed=False, reasoning=""),
        "efficiency": GraderResult(name="", score=0, passed=False, reasoning=""),
        "quality": GraderResult(name="", score=0, passed=False, reasoning=""),
        "claim_verification": GraderResult(name="", score=0, passed=False, reasoning=""),
        "classification": "",
        "average_score": 0.0,
    }

    result = compiled.invoke(initial_state)

    return AnalysisResult(
        run_id=trajectory.run_id,
        task=trajectory.task,
        classification=result["classification"],
        average_score=result["average_score"],
        grader_results=[
            result["task_completion"],
            result["efficiency"],
            result["quality"],
            result["claim_verification"],
        ],
        output=trajectory.output,
        tool_calls=[tc.model_dump() for tc in trajectory.tool_calls],
    )
```

- [ ] **Step 2: Verify imports**

```bash
PYTHONPATH=. python -c "from src.evolution.analyzer import analyze_trajectory, classify_trajectory; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add src/evolution/analyzer.py
git commit -m "refactor: rewrite analyzer to use evoagent graders and dataclass GraderResult

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Delete replaced src/ modules and update imports in remaining files

**Files:**
- Delete: `src/memory/store.py`, `src/memory/compression.py`
- Delete: `src/skills/manager.py`
- Delete: `src/evolution/skill_extractor.py`, `src/evolution/failure_skill_creator.py`
- Delete: `src/agent/middleware.py`, `src/agent/sleep_review.py`
- Delete: `src/tracing/trajectory.py`
- Modify: `src/memory/__init__.py`
- Modify: `src/skills/__init__.py`
- Modify: `src/tracing/__init__.py`
- Modify: `src/memory/reflection.py`
- Modify: `src/tracing/fetcher.py`

- [ ] **Step 1: Delete replaced files**

```bash
rm src/memory/store.py src/memory/compression.py
rm src/skills/manager.py
rm src/evolution/skill_extractor.py src/evolution/failure_skill_creator.py
rm src/agent/middleware.py src/agent/sleep_review.py
rm src/tracing/trajectory.py
```

- [ ] **Step 2: Update `src/memory/__init__.py`**

```python
"""Memory module — delegates to evoagent.memory."""
```

- [ ] **Step 3: Rewrite `src/memory/reflection.py`**

Replace `from src.memory.store import MemoryStore` with `from evoagent.memory.store import FileMemoryStore`:

```python
"""Post-run reflection generation."""

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from evoagent.memory.store import FileMemoryStore
from src.agent.prompts import REFLECTION_PROMPT

logger = logging.getLogger(__name__)


def generate_reflection(
    llm: BaseChatModel,
    task: str,
    output: str,
    tool_calls: list[dict[str, Any]],
    grader_results: dict[str, Any],
) -> dict[str, Any]:
    """Generate a reflection on an agent run using the LLM."""
    enriched_calls = []
    for tc in tool_calls[:50]:
        entry = {"name": tc.get("name", ""), "args": tc.get("args", {})}
        if "output" in tc:
            entry["output"] = str(tc["output"])[:300]
        enriched_calls.append(entry)

    prompt = REFLECTION_PROMPT.format(
        task=task,
        output=output[:8000],
        tool_calls=json.dumps(enriched_calls, indent=2),
        grader_results=json.dumps(grader_results, indent=2),
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        cleaned = response.content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines[1:] if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"summary": response.content[:500], "improvements": []}


def reflect_and_store(
    llm: BaseChatModel,
    memory_store: FileMemoryStore,
    run_id: str,
    task: str,
    output: str,
    tool_calls: list[dict[str, Any]],
    grader_results: dict[str, Any],
) -> dict[str, Any]:
    """Generate reflection and store as episodic + semantic memories."""
    reflection = generate_reflection(llm, task, output, tool_calls, grader_results)

    memory_store.store("episodic", f"run-{run_id[:8]}", {
        "task": task,
        "score": grader_results.get("average_score", 0),
        "classification": grader_results.get("classification", "unknown"),
        "summary": reflection.get("summary", ""),
        "improvements": reflection.get("improvements", []),
    })

    for fact in reflection.get("facts_learned", []):
        if isinstance(fact, str) and fact.strip():
            fact_id = f"fact-{run_id[:8]}-{hash(fact) % 10000}"
            memory_store.store("semantic", fact_id, {"content": fact.strip()})

    return reflection
```

- [ ] **Step 4: Update `src/skills/__init__.py`**

```python
"""Skills module — delegates to evoagent.skills."""
```

- [ ] **Step 5: Update `src/tracing/__init__.py`**

```python
"""Tracing module — delegates to evoagent.tracing."""
```

- [ ] **Step 6: Rewrite `src/tracing/fetcher.py`**

Replace `Trajectory` with `TrajectoryRecord`, `TrajectoryMetrics` with evoagent version:

```python
"""LangSmith trace fetcher."""

import logging
from pathlib import Path
from typing import Any

from langsmith import Client

from evoagent.core.types import TrajectoryMetrics
from evoagent.tracing.trajectory import ToolCall, TrajectoryRecord
from src.config.settings import Settings

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
        runs = list(
            self.client.list_runs(project_name=self.project_name, limit=limit)
        )
        return [self._run_to_dict(r) for r in runs]

    def fetch_run(self, run_id: str) -> dict[str, Any] | None:
        try:
            run = self.client.read_run(run_id)
            return self._run_to_dict(run)
        except Exception as exc:
            logger.error("Failed to fetch run %s: %s", run_id, exc)
            return None

    def _run_to_dict(self, run: Any) -> dict[str, Any]:
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
        logger.info("Skipping LangSmith trace fetch; using local trajectories")
        return []
```

- [ ] **Step 7: Verify all updated imports**

```bash
PYTHONPATH=. python -c "
from src.memory.reflection import reflect_and_store
from src.tracing.fetcher import TraceFetcher
print('All OK')
"
```

- [ ] **Step 8: Commit**

```bash
git add -A src/memory/ src/skills/ src/tracing/ src/agent/middleware.py src/agent/sleep_review.py src/evolution/skill_extractor.py src/evolution/failure_skill_creator.py
git commit -m "refactor: delete src/ modules replaced by evoagent, update remaining imports

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Rewrite orchestrator and supporting files

**Files:**
- Modify: `src/evolution/orchestrator.py`
- Modify: `src/evolution/evolution_state_bridge.py`
- Modify: `src/evolution/prompt_optimizer.py`
- Modify: `src/agent/deep_agent.py`
- Modify: `src/cli/commands.py`

- [ ] **Step 1: Rewrite `src/evolution/evolution_state_bridge.py`**

Replace `from src.evolution.state import AnalysisResult, EvolutionMetrics` (already works since state.py was updated in Task 1). Replace `from src.skills.manager import discover_skills` with `SkillManager`:

Change line: `from src.skills.manager import discover_skills` to `from evoagent.skills.manager import SkillManager`

Change `discover_skills(skills_dir)` calls to `SkillManager(skills_dir).discover()`.

Change all dict access on GraderResult (`grader["score"]`, `grader["passed"]`, `grader["name"]`, `grader["reasoning"]`) to attribute access (`grader.score`, `grader.passed`, `grader.name`, `grader.reasoning`).

Change all dict access on EvolutionMetrics (`m["cycle"]`, `m["avg_score"]`, etc.) — these remain TypedDicts so dict access stays.

- [ ] **Step 2: Rewrite `src/evolution/prompt_optimizer.py`**

Replace:
- `from src.agent.middleware import TRACES_DIR` → `TRACES_DIR = Path("traces")` (inline constant)
- `from src.evolution.state import AnalysisResult` → keep (state.py updated)
- `from src.skills.manager import discover_skills` → `from evoagent.skills.manager import SkillManager`

Change `discover_skills(skills_dir)` to `SkillManager(skills_dir).discover()`.

Change GraderResult dict access in failure analysis to attribute access where GraderResult objects are accessed.

- [ ] **Step 3: Rewrite `src/evolution/orchestrator.py`**

This is the biggest change. Replace:
- `from src.memory.store import MemoryStore` → `from evoagent.memory.store import FileMemoryStore`
- `from src.memory.compression import consolidate_episodic, deduplicate_semantic` → `from evoagent.memory.compression import deduplicate_semantic`
- `from src.evolution.skill_extractor import extract_skills_from_batch` → `from evoagent.skills.extractor import extract_skills_from_batch`
- `from src.evolution.failure_skill_creator import create_failure_skills_from_batch` → `from evoagent.skills.extractor import extract_skills_from_batch` (use mode="failure")
- `from src.tracing.trajectory import Trajectory` → `from evoagent.tracing.trajectory import TrajectoryRecord`
- `from src.tracing.trajectory import TrajectoryMetrics` → `from evoagent.core.types import TrajectoryMetrics`
- `from src.skills.manager import discover_skills` → `from evoagent.skills.manager import SkillManager`

Change `MemoryStore` to `FileMemoryStore` throughout.
Change `Trajectory(...)` to `TrajectoryRecord(...)` — note field mapping: old `metrics=TrajectoryMetrics(...)` → new flat fields `total_tokens=`, `total_steps=`, etc.
Change `create_failure_skills_from_batch(llm, analyses, skills_dir)` → `extract_skills_from_batch(llm, analyses_as_dicts, SkillManager(skills_dir), mode="failure")`.
Change GraderResult dict access to attribute access in logging/metrics.

- [ ] **Step 4: Rewrite `src/agent/deep_agent.py`**

Replace:
- `from src.agent.middleware import (...)` → `from evoagent.harness.middleware import (...)`
- `from src.memory.compression import compress_memory_context` → `from evoagent.memory.compression import compress_context`
- `from src.memory.store import MemoryStore` → `from evoagent.memory.store import FileMemoryStore`

Change `MemoryStore` to `FileMemoryStore` in function signatures.
Change `compress_memory_context(...)` to `compress_context(...)`.

- [ ] **Step 5: Rewrite `src/cli/commands.py`**

Replace:
- `from src.memory.store import MemoryStore` → `from evoagent.memory.store import FileMemoryStore`
- `from src.skills.manager import list_skills` → `from evoagent.skills.manager import SkillManager`
- `from src.agent.sleep_review import run_sleep_review` → `from evoagent.evolution.sleep_review import run_sleep_review`

Change `MemoryStore(...)` → `FileMemoryStore(...)`.
Change `list_skills(skills_dir)` → implement inline using `SkillManager(skills_dir).discover()`.

- [ ] **Step 6: Verify full import chain**

```bash
PYTHONPATH=. python -c "
from src.evolution.orchestrator import run_evolution
from src.cli.commands import main
from src.agent.deep_agent import create_agent
print('Full import chain OK')
"
```

- [ ] **Step 7: Commit**

```bash
git add src/
git commit -m "refactor: rewrite orchestrator, deep_agent, CLI to use evoagent library

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Update test_graders.py and run full validation

**Files:**
- Modify: `test_graders.py`

- [ ] **Step 1: Update `test_graders.py`**

Replace imports:
- `from src.evolution.graders.multi_judge import ...` → `from evoagent.graders.multi_judge import MultiJudgeGrader`
- `from src.evolution.graders.efficiency import grade_efficiency` → `from evoagent.graders.efficiency import EfficiencyGrader`
- `from src.tracing.trajectory import TrajectoryMetrics` → `from evoagent.core.types import TrajectoryMetrics`
- `from src.evolution.graders.claim_verification import ...` → keep (still in src since it's research-specific)
- `from src.evolution.graders.fact_checker import ...` → keep
- `from src.evolution.analyzer import classify_trajectory` → keep

Update grader calls to use class API.
Update GraderResult access from dict to attribute style.

- [ ] **Step 2: Run all existing tests**

```bash
PYTHONPATH=. python -m pytest tests/ -x -q 2>&1
```

Fix any failures.

- [ ] **Step 3: Run evoagent tests to confirm no regression**

```bash
PYTHONPATH=. python -m pytest tests/evoagent/ -q
```

Expected: 52 passed.

- [ ] **Step 4: Commit**

```bash
git add test_graders.py
git commit -m "refactor: update test_graders.py to use evoagent imports

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```
