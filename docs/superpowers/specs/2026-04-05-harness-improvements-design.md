# Harness Layer Improvements Design Spec

**Date:** 2026-04-05
**Status:** Approved

## Problem

Comparison with Meta-Harness (Stanford), LangChain Harness Engineering, and Karpathy's autoresearch identified 7 gaps in our harness layer. All gaps have proven impact in their source systems.

## Improvements

### 1. Full Environment Bootstrapping

**Source:** Meta-Harness, LangChain (`LocalContextMiddleware`)
**Impact:** Eliminates 2-5 wasted exploration turns

Enhance `ContextAssemblyMiddleware` to inject a system environment snapshot on the first model call. Opt-in via `detect_environment: bool = False`.

When enabled, the injected context includes:
- Working directory path
- Directory structure (capped at 50 entries, 2 levels deep)
- Available tools (runs `which` on fixed list: python3, node, curl, git, etc.)
- System memory available

No new class — extends the existing `ContextAssemblyMiddleware` constructor.

### 2. Time Budget Warnings

**Source:** LangChain
**Impact:** Prevents timeout failures

New middleware: `TimeBudgetMiddleware(budget_seconds=300, warn_at=[0.6, 0.85])`.

Implemented via `wrap_tool_call` — appends warnings to tool responses:
- At 60% elapsed: `"Start synthesizing your findings."`
- At 85% elapsed: `"Stop all searches. Write your final report NOW."`
- Each warning fires only once
- Zero extra LLM calls

### 3. File-Edit Loop Detection

**Source:** LangChain (`LoopDetectionMiddleware`)
**Impact:** Catches edit-retry doom loops

Extend existing `LoopDetectionMiddleware` with two new tracking dimensions:
- `max_file_edits=5`: warn after N edits to the same file
- `max_repeated_tools=4`: warn after N calls to same tool with similar args (>60% overlap)

Tracked via `wrap_tool_call`. Extracts file paths from tool args. Non-binding warnings.

### 4. Parallel Error Analysis Agents

**Source:** LangChain (Trace Analyzer Skill), Meta-Harness (counterfactual diagnosis)
**Impact:** Deeper failure diagnosis for prompt optimizer

New module: `evoagent/evolution/error_analyzer.py`.

```python
def analyze_failures_deep(
    llm, failed_analyses, traces_dir=None, max_parallel=3
) -> list[dict]:
```

For each failure, spawns a parallel LLM call (via `ThreadPoolExecutor(max_workers=3)`) that:
1. Reads the execution trace
2. Identifies the exact decision point where the agent went wrong
3. Proposes what the prompt should have said instead

Returns `list[{task, root_cause, counterfactual, suggested_guidance}]`.

Output feeds into prompt optimizer's metaprompt as `{error_analysis}` section, replacing shallow issue list. Falls back to current shallow aggregation if not available.

Both `evoagent/evolution/prompt_optimizer.py` and `src/evolution/prompt_optimizer.py` updated.

### 5. Reasoning Compute Allocation

**Source:** LangChain ("Reasoning Sandwich", +2.9 pts)
**Impact:** Better planning and verification without timeout

New middleware: `ReasoningSandwichMiddleware(planning_effort="high", implementation_effort="medium", verification_effort="high", planning_calls=2)`.

Implemented via `before_model`:
- Calls 1-2 (planning): high reasoning effort
- Calls 3+ (implementation): medium reasoning effort
- After 60% time elapsed (verification phase): switches back to high

Provider-aware: only injects `reasoning_effort` / `thinking` params for models that support it. No-op for others.

### 6. Pre-Completion Verification Checklist

**Source:** LangChain (`PreCompletionChecklistMiddleware`)
**Impact:** Catches logical errors, not just structural

Enhance `SelfVerificationMiddleware` with `verify_against_task: bool = False`.

When enabled, after structural checks pass:
1. Extract original task from first HumanMessage
2. Lightweight LLM call: "Does this output fully address the task?"
3. If not addressed: inject revision prompt listing missing aspects

Cost: 1 extra LLM call per completion attempt (max 3 with retries). Structural checks gate the LLM call — no LLM cost if structure already fails.

Requires passing `llm` to the middleware constructor when `verify_against_task=True`.

### 7. Memory Contradiction Detection

**Source:** Letta (Continual Learning)
**Impact:** Prevents stale/wrong memories from degrading performance

Enhance `run_sleep_review()` with `detect_contradictions: bool = True, max_memories_to_scan: int = 50`.

Process:
1. Load recent semantic memories (up to cap)
2. Cluster by topic similarity (reuse `_jaccard_similarity`)
3. For each cluster with 2+ items: LLM call to detect contradictions
4. Resolve: keep newer/higher-confidence memory, delete the other

Cost: ~3-5 extra LLM calls per sleep review (one per cluster).

## Files Modified

| File | Changes |
|------|---------|
| `evoagent/harness/middleware.py` | Enhance ContextAssembly (env snapshot), enhance LoopDetection (file edits), enhance SelfVerification (task verification), add TimeBudgetMiddleware, add ReasoningSandwichMiddleware |
| `evoagent/harness/builder.py` | Update `default_middleware_stack()` with new middleware + params |
| `evoagent/evolution/error_analyzer.py` | New: parallel deep failure analysis |
| `evoagent/evolution/prompt_optimizer.py` | Use `analyze_failures_deep()` in metaprompt |
| `evoagent/evolution/sleep_review.py` | Add contradiction detection pass |
| `src/agent/deep_agent.py` | Wire new middleware params (time budget, env detection, task verification) |
| `src/evolution/prompt_optimizer.py` | Use `analyze_failures_deep()` |
| Tests | New tests for each improvement |

## What Does NOT Change

- Core protocols (Grader, AgentFactory, etc.)
- Core types (GraderResult, TaskResult, etc.)
- Memory store, skills manager, tracing
- Orchestrator pipeline structure
- Grader implementations
