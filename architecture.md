# Architecture: Auto-Research Self-Improving Agents

This document describes the architecture of the self-improving research agent system, built on the **`evoagent`** reusable library. It includes a comprehensive comparison with the four research systems that informed its design.

---

## Research Foundations

| Source | Key Idea | What We Adopted | What We Changed |
|--------|----------|-----------------|-----------------|
| [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) | Greedy hill-climbing: AI proposes code changes, 5-min eval, accept/reject | Outer-loop structure, experiment tracking, accept/reject pattern | Made it continuous via daemon instead of batch overnight runs |
| [LangChain Harness Engineering](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) | Agent = Model + Harness; middleware stack (verification, loop detection, time budget, reasoning sandwich) | All 6 middleware classes, trace-driven optimization, reasoning sandwich | Added auto-tuning of middleware parameters; their approach is manual |
| [Stanford Meta-Harness](https://yoonholee.com/meta-harness/) | Automated harness search with 10M tokens of raw trace context; optimizes full harness (prompts + tools + completion logic) | Tool config optimization, completion check evolution, multi-candidate search, holdout evaluation | We optimize continuously from real interactions; they run fixed-budget batch search |
| [Letta Continual Learning](https://www.letta.com/blog/continual-learning) | Token-space learning: update context not weights; sleep-time compute; skill learning | Episodic/semantic memory, sleep-time review, skill extraction, memory compression | Added background daemon that learns from every interaction without explicit triggers |
| [Harrison Chase: Agent Anatomy](https://blog.langchain.com/the-anatomy-of-an-agent-harness/) | Three learning surfaces: model, harness, context; harness-model co-evolution | Versioned harness config alongside versioned prompts; separate optimization for each surface | Unified daemon optimizes all three surfaces from a single run log |

---

## Comprehensive Comparison

### Architecture Overview

```
OUR SYSTEM                          META-HARNESS (Stanford)
┌──────────────┐                    ┌──────────────┐
│ User runs    │──→ run_log.jsonl   │ Proposer     │──→ filesystem
│ (real tasks) │                    │ (Claude Code)│    (10M tokens)
└──────┬───────┘                    └──────┬───────┘
       │ poll 60s                          │ iterate
       ▼                                   ▼
┌──────────────┐                    ┌──────────────┐
│ Daemon       │                    │ Evaluator    │
│ 6 steps:     │                    │ (run tasks)  │
│ 1.Prompt opt │                    └──────┬───────┘
│ 2.Skills     │                           │
│ 3.Fail skills│                    ┌──────▼───────┐
│ 4.Memory     │                    │ Filesystem   │
│ 5.Harness opt│                    │ (all traces, │
│ 6.Tool config│                    │  all code)   │
└──────┬───────┘                    └──────────────┘
       │ writes
       ▼
  prompts/ + harness_config/ + skills/

LANGCHAIN                           AUTORESEARCH (Karpathy)
┌──────────────┐                    ┌──────────────┐
│ Trace        │                    │ AI Agent     │
│ Analyzer     │──→ LangSmith      │ (Claude/     │
│ Skill        │    traces          │  Codex)      │
└──────┬───────┘                    └──────┬───────┘
       │ findings                          │ modify train.py
       ▼                                   ▼
┌──────────────┐                    ┌──────────────┐
│ Human        │                    │ 5-min eval   │
│ Engineer     │                    │ (val_bpb)    │
│ (applies     │                    └──────┬───────┘
│  changes)    │                           │
└──────────────┘                    keep or discard
```

### What Gets Optimized

| Component | **Our System** | **Meta-Harness** | **LangChain** | **Autoresearch** |
|---|---|---|---|---|
| System prompt | Yes — multi-candidate LLM rewriting with dimension breakdown | Yes — proposer rewrites freely | Yes — manually tuned | N/A |
| Tool definitions | **Yes** — search retry, delay, depth, max_results via HarnessConfig | Yes — proposer rewrites tool schemas | No (fixed) | N/A |
| Completion logic | **Yes** — LLM-evaluated custom checks evolve via harness optimizer | Yes — proposer modifies stop logic | Yes — PreCompletionChecklist (manual) | N/A |
| Middleware parameters | **Yes** — retry counts, time budgets, loop thresholds, reasoning effort | Implicit (part of full harness rewrite) | Manually tuned | N/A |
| Skills / patterns | **Yes** — auto-extracted from successes + defensive from failures | No | No | N/A |
| Memory / context | **Yes** — episodic + semantic, sleep-time compute, compression | No | No | N/A |
| Code / architecture | No (outer loop, manual) | No | No | **Yes** — primary target |

### Optimization Trigger and Lifecycle

| | **Our System** | **Meta-Harness** | **LangChain** | **Autoresearch** |
|---|---|---|---|---|
| **Trigger** | Automatic — daemon polls every 60s, fires after 3+ user runs | Explicit — research loop with fixed iteration budget | Manual — engineer reads trace analysis | Explicit — agent proposes, human starts |
| **Eval source** | Real user interactions (diverse) | Held-out benchmark tasks (controlled) | TerminalBench 2.0 (89 tasks) | Fixed validation dataset |
| **Eval metric** | 4-dimension grading (task_completion, efficiency, quality, claims) | Binary pass/fail (5 trials per task) | Binary pass/fail | val_bpb (continuous) |
| **Overfitting risk** | Low (real usage = diverse) | Medium (fixed held-out set) | Low (large benchmark) | Medium (fixed val set) |
| **Lifecycle** | Continuous — never stops while daemon runs | Fixed budget — 20 iterations then stops | Manual iteration — human decides | Overnight batch — ~100 experiments |
| **Learning persistence** | Permanent — skills, memory, prompt versions, harness configs accumulate | None — each optimization run is independent | None | None |

### Diagnostic Context for Optimization

| | **Our System** | **Meta-Harness** | **LangChain** | **Autoresearch** |
|---|---|---|---|---|
| **Volume** | ~5-20K tokens (LangSmith traces + grading) | **~10M tokens** (full filesystem) | Full LangSmith traces | Previous code + score |
| **Access method** | LangSmith API → formatted text; local traces as fallback | Unix commands (grep, cat) on raw logs | LangSmith UI + parallel analysis agents | Agent reads code diff |
| **Trace depth** | All descendant runs via `list_runs(trace_id=...)`: tool args/results, LLM decisions, errors | Raw execution logs at any depth | Full LangSmith traces | N/A |
| **Per-dimension analysis** | Yes — failure rates, bottleneck identification, grader reasoning | No — binary pass/fail only | No — binary pass/fail | No — single scalar metric |

### Search Strategy

| | **Our System** | **Meta-Harness** | **LangChain** | **Autoresearch** |
|---|---|---|---|---|
| **Candidates per cycle** | 2 (prompt) + 2 (harness) | 2 per iteration | 1 (manual) | 1 |
| **Selection method** | Prompt: least length drift; Harness: most conservative (fewest changes) | Filesystem-based scoring + proposer judgment | Human judgment | Score comparison |
| **Validation** | Pairwise comparison on 2 failed tasks | Held-out task evaluation | Full benchmark rerun | val_bpb comparison |
| **Rollback** | Versioned stores with incremental scoring | Filesystem preserves all candidates | Git revert | Automatic discard |
| **Safety** | Parameter bounds clamping, autonomy violation detection, line stripping | Human review | Human review | Automatic accept/reject |

### Results and Scale

| | **Our System** | **Meta-Harness** | **LangChain** | **Autoresearch** |
|---|---|---|---|---|
| **Reported results** | Prompt v7→v12, daemon confirmed working, harness optimization active | Claude Opus 4.6 → 76.4% TBench-2 (rank #2) | 52.8% → 66.5% TBench-2 (+13.7 pts) | Incremental val_bpb improvements |
| **Optimization cost** | Low — no task re-running, uses existing grading results | High — Claude Code as proposer + full task evaluation | Medium — human time + benchmark reruns | Medium — GPU compute per experiment |
| **Scale tested** | Research prototype (real user interactions) | 89-task benchmark, multiple domains | 89-task benchmark | Single GPU, overnight |

---

## What We Have That Nobody Else Does

### 1. Continuous Background Evolution
Every other system requires an explicit optimization trigger. Our daemon learns from every user interaction automatically. The agent improves while you use it — no batch runs, no manual intervention, no overnight sessions.

### 2. Persistent Skill Library
No other system accumulates reusable patterns across sessions. Our skill extractor creates SKILL.md files from both successes (reusable strategies) and failures (defensive "antibody" skills). These are injected into every future run via ContextAssemblyMiddleware.

### 3. Episodic + Semantic Memory with Sleep-Time Compute
Letta proposed this architecture; we implemented it. Episodic memories capture per-run experiences. Semantic memories extract generalizable facts and patterns. Sleep-time review identifies cross-cutting patterns and generates meta-instructions. Token-budgeted compression prevents context bloat.

### 4. Unified Harness + Prompt + Tool Optimization
Meta-Harness optimizes the harness. Autoresearch optimizes code. LangChain optimizes the prompt manually. Our daemon optimizes all three — system prompt, middleware parameters (6 middleware classes with ~20 tunable parameters), tool configurations (search retry/depth/results), and custom completion checks — from a single optimization cycle.

### 5. Multi-Dimensional Grading with Critical Grader Gate
Every other system uses binary pass/fail or a single scalar metric. Our 4-dimension grading (task_completion, efficiency, quality, claim_verification) with per-dimension failure analysis gives the optimizer precise diagnostic signal. The critical grader gate ensures task_completion failures always trigger optimization, even when other dimensions score well.

### 6. Incremental Scoring from Every Interaction
Prompt versions and harness configs accumulate scores from every single user run via incremental averaging. After 10 runs, the system has a statistically reliable score for each version — far more signal than Meta-Harness's held-out evaluation or Autoresearch's single-metric comparison.

---

## Remaining Gaps vs. Reference Systems

### vs. Meta-Harness
| Gap | Impact | Mitigation |
|-----|--------|------------|
| **Diagnostic context volume** — they give 10M tokens, we give 5-20K | Their optimizer can do deeper counterfactual diagnosis | We use LangSmith traces with full descendant run visibility; could increase to 200K tokens with larger context models |
| **Agentic optimizer** — their proposer uses Claude Code with filesystem tools | They can grep raw logs and trace failures to specific code lines | Our optimizer is a single LLM call; could be upgraded to a tool-using agent |
| **Code modification** — their proposer can rewrite harness source code | They can make structural changes we can't | Our outer loop (coding agent) handles this manually when the daemon plateaus |

### vs. LangChain
| Gap | Impact | Mitigation |
|-----|--------|------------|
| **Parallel trace analysis agents** — they spawn multiple agents to analyze traces | Deeper, more diverse failure analysis | Our multi-candidate generation partially compensates; could add parallel analysis |

### vs. Autoresearch
| Gap | Impact | Mitigation |
|-----|--------|------------|
| **Automated code modification** — they modify train.py autonomously | Can discover novel architectures and training strategies | Our outer loop is manual; automating it with safety constraints is the next major milestone |

---

## Pros and Cons of Our Approach

### Pros

| Advantage | Why It Matters |
|-----------|---------------|
| **Continuous learning from real usage** | No synthetic benchmark overfitting. The agent improves on the actual distribution of tasks users care about, not a fixed test set. |
| **Zero-cost optimization** | The daemon never re-runs tasks. It works entirely on grading results already collected, making each optimization cycle cheap (a few LLM calls for proposal generation). |
| **Full-stack optimization** | Prompt + harness parameters + tool config + completion logic + skills + memory — all from a single daemon cycle. No other system optimizes this many surfaces. |
| **Persistent learning** | Skills, memories, and prompt versions accumulate across sessions. The agent genuinely gets better over time, not just within a single optimization run. |
| **Safe parameter evolution** | All numeric parameters are clamped to safe bounds. Effort strings are validated. Autonomy violations are detected and stripped. Multi-candidate selection picks the most conservative proposal. The system can't drift into a broken state. |
| **Graceful degradation** | LangSmith unavailable? Falls back to local traces. LLM returns garbage? Falls back to current config. All candidates fail validation? Keeps current version. Every optimization step has a safe fallback. |
| **Observable and debuggable** | Versioned JSON files for prompts and harness configs. SKILL.md files are human-readable. Run log is append-only JSONL. Every decision is traceable. |
| **Incremental scoring** | Every user interaction contributes to version scores. After enough runs, `get_current_prompt()` and `load_best()` return statistically reliable winners — more signal than any batch evaluation. |

### Cons

| Limitation | Why It Matters | Potential Fix |
|-----------|---------------|---------------|
| **Shallow diagnostic context** | 5-20K tokens vs Meta-Harness's 10M means our optimizer can't do deep counterfactual diagnosis — it sees summaries, not raw evidence | Increase context to 200K with larger models, or upgrade to an agentic optimizer with file-reading tools |
| **Single LLM call for optimization** | The prompt and harness optimizers each make one LLM call (x2 candidates). Meta-Harness's proposer is a full coding agent that can iterate | Replace with a LangGraph agent that has tools: `read_trace`, `grep_traces`, `read_config_history` |
| **No automated code modification** | Can't add new tools, change model, or restructure the agent architecture — requires manual outer loop | Automate the outer loop with a constrained coding agent (scope to specific files, require tests to pass) |
| **Cold start problem** | With no run history, the daemon has nothing to optimize. The system needs initial user interactions or bootstrap runs before self-improvement kicks in | Optional `evolve` bootstrapping command exists for this; could also seed with synthetic tasks on first launch |
| **Memory growth** | Semantic memories grow ~100 per cycle. Deduplication helps but doesn't prevent gradual context bloat | More aggressive compression, or switch to vector-based retrieval instead of keyword search |
| **Grading cost** | 4-dimension grading requires ~4 LLM calls per run (multi-judge). This doubles the cost of each user interaction | Could use a cheaper model for grading, or grade asynchronously after returning the result to the user |
| **Single-machine daemon** | The daemon runs as a single process. No distributed coordination, no horizontal scaling | For a research prototype this is fine; production would need a proper job queue |
| **Conservative selection bias** | Multi-candidate selection picks the most conservative harness proposal (fewest changes) and least-drift prompt. This prevents catastrophic changes but may also prevent bold improvements | Could alternate between conservative and exploratory selection strategies across cycles |

---

## Continuous Self-Improvement Model

The system improves through three concurrent modes:

```
┌───────────────────────────────────────────────────────────────┐
│  OUTER LOOP (structural changes, manual)                       │
│  Coding agent reads evolution_state/ → new tools, model swaps  │
│  Triggered when daemon plateaus                                │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  BACKGROUND DAEMON (continuous, primary evolution path)    │  │
│  │  Watches run_log.jsonl → when 3+ runs accumulate:        │  │
│  │  1. Prompt optimization (2 candidates, least-drift)      │  │
│  │  2. Skill extraction (success patterns)                  │  │
│  │  3. Failure skill creation (defensive patterns)          │  │
│  │  4. Memory compression (semantic dedup)                  │  │
│  │  5. Harness optimization (2 candidates, most conservative)│  │
│  │     - Middleware params, tool config, completion checks   │  │
│  │  Uses LangSmith traces for counterfactual diagnosis      │  │
│  │                                                          │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  HARNESS (per-step, within each run)               │  │  │
│  │  │  6 middleware classes, all params evolvable:        │  │  │
│  │  │                                                    │  │  │
│  │  │  SelfVerification → errors, sections, custom checks │  │  │
│  │  │  ContextAssembly  → env detection, skills injection │  │  │
│  │  │  LoopDetection    → query/edit/tool loop breaking   │  │  │
│  │  │  TimeBudget       → soft time pressure warnings     │  │  │
│  │  │  ReasoningSandwich → phased reasoning effort         │  │  │
│  │  │  TraceCapture     → records for LangSmith + local   │  │  │
│  │  │                                                    │  │  │
│  │  │  ┌──────────────────────────────────────────────┐  │  │  │
│  │  │  │  AGENT (per-task)                            │  │  │  │
│  │  │  │  Think → Search → Synthesize → Write report  │  │  │  │
│  │  │  │  Post-run: grade → score → reflect → log     │  │  │  │
│  │  │  └──────────────────────────────────────────────┘  │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  SLEEP-TIME COMPUTE (between sessions)                         │
│  Cross-run trace analysis → meta-instructions → memory         │
└───────────────────────────────────────────────────────────────┘
```

### How each mode feeds back

| Mode | Trigger | What it improves | Timescale |
|------|---------|-----------------|-----------|
| Per-run learning | Every `python -m src run` | Prompt score, harness score, failure feedback, episodic/semantic memory | Seconds |
| Background daemon | 3+ unprocessed runs in log | Prompt, harness config, tool config, completion checks, skills, memory | Minutes |
| Batch evolution | `python -m src evolve` (bootstrap only) | Same as daemon, plus holdout generalization check | Minutes-hours |
| Sleep-time review | `python -m src sleep-review` | Meta-instructions, cross-run patterns | Between sessions |
| Outer loop | Coding agent on plateau | Tools, model, architecture, grading pipeline | Hours-days |

---

## Two-Package Architecture

```
evoagent/                    src/
(generic patterns)           (research-specific application)

core/types.py ◄──────────── evolution/state.py (imports GraderResult)
core/protocols.py ◄──────── agent/deep_agent.py (implements AgentFactory pattern)
core/parsing.py ◄──────────  evolution/graders/*.py (uses parse_llm_json)
memory/store.py ◄──────────  cli/commands.py, orchestrator.py, daemon.py
memory/compression.py ◄────  orchestrator.py, daemon.py (deduplicate_semantic)
skills/manager.py ◄────────  orchestrator.py, daemon.py, CLI
skills/extractor.py ◄──────  orchestrator.py, daemon.py
harness/middleware.py ◄────  agent/deep_agent.py (6 middleware classes)
harness/builder.py ◄───────  agent/deep_agent.py (default_middleware_stack + HarnessConfig)
graders/efficiency.py ◄────  evolution/analyzer.py
graders/multi_judge.py ◄───  evolution/analyzer.py
evolution/analyzer.py ◄────  evolution/analyzer.py (classify_trajectory + critical gate)
evolution/sleep_review.py ◄  cli/commands.py
tracing/trajectory.py ◄────  evolution/state.py
```

---

## EvoAgent Library (`evoagent/`)

### Layer 0: Core (pydantic only)

**`core/types.py`** — Shared data types: `GraderResult`, `TaskResult`, `TrajectoryMetrics`, `EvolutionCycleReport`

**`core/protocols.py`** — ABCs: `Grader`, `AgentFactory`, `MemoryBackend`, `SkillStore`, `PromptStore`

**`core/config.py`** — `EvoAgentConfig` with evolution, memory, and directory defaults

**`core/parsing.py`** — `parse_llm_json()` handles markdown blocks, trailing text, fallback

### Layer 1: Memory & Skills (core only)

**`memory/store.py`** — `FileMemoryStore`: episodic/semantic namespaces, JSON per item, keyword search

**`memory/compression.py`** — `compress_context()` (token-budgeted, priority-ordered) + `deduplicate_semantic()` (Jaccard similarity)

**`skills/manager.py`** — `SkillManager`: SKILL.md files with YAML frontmatter, progressive disclosure

**`skills/extractor.py`** — Success (score >= 0.70) + failure (score < 0.50) skill extraction with dedup

### Layer 2: Harness & Graders (needs langchain-core)

**`harness/middleware.py`** — Six middleware classes:

| Middleware | Hook | Purpose | Evolvable Parameters |
|-----------|------|---------|---------------------|
| `SelfVerificationMiddleware` | `after_model` | Checks output structure, errors, custom completion checks | max_retries, min_length, required_sections, completion_checks, verify_against_task |
| `ContextAssemblyMiddleware` | `before_model` | Injects skills + environment context | detect_env |
| `LoopDetectionMiddleware` | `wrap_tool_call` | Breaks repetitive query/edit/tool loops | max_similar, max_total, max_file_edits, max_repeated_tools |
| `TimeBudgetMiddleware` | `wrap_tool_call` | Soft time pressure warnings | budget_seconds, warn_at |
| `ReasoningSandwichMiddleware` | `before_model` | Phased reasoning effort allocation | planning_effort, implementation_effort, verification_effort, planning_calls |
| `TraceCaptureMiddleware` | `before/after_agent` | Records traces to disk + LangSmith | (not tunable) |

**`harness/builder.py`** — `default_middleware_stack(harness_config=...)` constructs all 6 from HarnessConfig

**`graders/efficiency.py`** — Rule-based: 50% tokens + 30% steps + 20% latency

**`graders/multi_judge.py`** — 3 parallel LLM judges, median aggregation, `[low_agreement]` flag

### Layer 3: Evolution (needs langgraph)

**`evolution/analyzer.py`** — `classify_trajectory()` with critical grader gate: `CRITICAL_GRADERS = {"task_completion"}`

**`evolution/sleep_review.py`** — Cross-run trace analysis, meta-instruction generation

---

## Research Agent Application (`src/`)

### Agent Core

**`deep_agent.py`** — Loads best prompt (PromptStore) + best harness config (HarnessConfigStore) + memory context, wires middleware stack, creates agent

**`prompt_store.py`** — Versioned prompts with incremental scoring, feedback accumulation

**`tools/search.py`** — Tavily + DuckDuckGo fallback; retry/depth/results configurable via HarnessConfig

### Evolution Pipeline

**`orchestrator.py`** — 10-node LangGraph pipeline (bootstrap only):
```
START → run_batch → analyze → reflect → compress_memories
    → extract_skills → create_failure_skills → optimize_prompt
    → holdout_check → persist_state → aggregate_metrics → [continue/END]
```

**`analyzer.py`** — 4-axis grading via parallel fan-out with critical grader gate

**`prompt_optimizer.py`** — Multi-candidate (n=2), dimension-aware, autonomy-safe, LangSmith-trace-informed

**`harness_config.py`** — `HarnessConfig` dataclass (~20 tunable params) + `HarnessConfigStore` (versioned JSON, incremental scoring)

**`harness_optimizer.py`** — Multi-candidate (n=2, most conservative wins), parameter bounds clamping, LangSmith traces

**`daemon.py`** — Background daemon: polls run_log.jsonl, triggers 5-step evolution cycle

**`run_log.py`** — Append-only JSONL with thread-safe writes, processed tracking

### Tracing

**`tracing/fetcher.py`** — `TraceFetcher` fetches rich LangSmith traces via `list_runs(trace_id=...)` for full descendant visibility. Falls back to local trace files.

---

## Data Flow: Single Run → Background Evolution

```
User: python -m src run "Research fusion energy"
         │
         ▼
┌──────────────────────────────────┐
│  Agent runs (6 middleware active) │
│  Best prompt + best harness cfg  │
│  Output: research report         │
└──────────────┬───────────────────┘
               │
         ┌─────┴──────┐
         ▼            ▼
┌──────────────┐ ┌──────────────────┐
│  Grade (4x)  │ │  Print to user   │
│  TC, Eff,    │ └──────────────────┘
│  Quality,    │
│  Claims      │
└──────┬───────┘
       │
  ┌────┴────┬──────────┬──────────┬──────────────┐
  ▼         ▼          ▼          ▼              ▼
Score     Score      Feedback   Reflect        Append to
prompt    harness    on fails   → memory       run_log.jsonl
version   config     → prompt                       │
                     version                        │ (daemon polls)
                                                    ▼
                                          ┌──────────────────┐
                                          │ Background Daemon │
                                          │ (when 3+ pending) │
                                          └────────┬─────────┘
                                                   │
                             ┌──────────┬──────────┼──────────┬──────────┐
                             ▼          ▼          ▼          ▼          ▼
                        Optimize   Optimize    Extract    Extract    Compress
                        prompt     harness     skills     fail       memory
                        (2 cand)   (2 cand)              skills
                             │          │          │
                             ▼          ▼          ▼
                       prompts/   harness_config/ skills/
                             │          │
                             ▼          ▼
                    Next run picks up improved
                    prompt AND evolved harness
```

---

## Critical Thresholds

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `CRITICAL_GRADERS` | `{"task_completion"}` | Both `analyzer.py` | Caps classification at partial when failing |
| `SUCCESSFUL_THRESHOLD` | 0.75 | `evoagent/evolution/analyzer.py` | Classification as successful |
| `PARTIAL_THRESHOLD` | 0.50 | `evoagent/evolution/analyzer.py` | Classification as partial |
| `MIN_PASS_COUNT` | 2 | `evoagent/evolution/analyzer.py` | Graders that must pass |
| `SUCCESS_THRESHOLD` | 0.70 | `evoagent/skills/extractor.py` | Min score for success skills |
| `FAILURE_THRESHOLD` | 0.50 | `evoagent/skills/extractor.py` | Max score for failure skills |
| `MAX_SKILLS_PER_CYCLE` | 3 | `evoagent/skills/extractor.py` | Prevent skill explosion |
| `MIN_IMPROVEMENT` | 0.05 | `src/evolution/orchestrator.py` | Plateau detection |
| `HOLDOUT_FRACTION` | 0.30 | `src/evolution/orchestrator.py` | Tasks reserved for holdout |
| `n_candidates` | 2 | prompt_optimizer + harness_optimizer | Candidates per optimization cycle |
| `DEFAULT_POLL_INTERVAL` | 60s | `src/evolution/daemon.py` | Daemon polling interval |
| `DEFAULT_MIN_RUNS` | 3 | `src/evolution/daemon.py` | Runs before daemon triggers |
| `_PARAM_BOUNDS` | varies | `src/evolution/harness_optimizer.py` | Safe bounds for all tunable params |

---

## Technology Stack

| Component | Technology | Package |
|-----------|-----------|---------|
| Core types & protocols | `pydantic`, `pydantic-settings` | `evoagent` (core) |
| Memory & skills | File-backed JSON, SKILL.md | `evoagent` (core) |
| Middleware | `AgentMiddleware` (6 classes) | `evoagent[harness]` |
| Graders | `ThreadPoolExecutor`, LLM-as-judge | `evoagent[graders]` |
| Evolution loop | `langgraph` StateGraph | `evoagent[evolution]` |
| Agent framework | `deepagents` | `src/` app |
| LLM abstraction | `langchain` + `init_chat_model()` | `src/` app |
| Web search | `langchain-tavily` + `ddgs` | `src/` app |
| Tracing | `langsmith` (read + write) | `src/` app |

---

## Persistence Layout

```
auto_research_self_improving_agents/
├── evoagent/                 # Reusable library
│   ├── core/                 # Types, protocols, config, parsing
│   ├── memory/               # FileMemoryStore, compression
│   ├── skills/               # SkillManager, extractor
│   ├── harness/              # 6 middleware classes + builder (HarnessConfig-aware)
│   ├── graders/              # EfficiencyGrader, MultiJudgeGrader
│   ├── evolution/            # Analyzer (critical grader gate), sleep review
│   └── tracing/              # TrajectoryRecord, ToolCall
│
├── src/                      # Research agent application
│   ├── agent/                # deep_agent (loads best prompt + harness config), prompts, prompt_store
│   ├── evolution/            # daemon, harness_config, harness_optimizer, prompt_optimizer,
│   │                         #   orchestrator (bootstrap), analyzer, run_log, graders/
│   ├── memory/               # reflection.py
│   ├── tools/                # search.py (configurable via HarnessConfig)
│   ├── tracing/              # fetcher.py (LangSmith rich traces + local fallback)
│   ├── config/               # settings.py
│   └── cli/                  # commands.py (run, evolve-daemon, evolve, etc.)
│
├── memory/                   # Runtime: episodic + semantic memories
├── skills/                   # Runtime: learned SKILL.md files
├── prompts/                  # Runtime: versioned prompts with scores
├── harness_config/           # Runtime: versioned harness configs with scores
├── traces/                   # Runtime: local execution traces
├── evolution_state/          # Runtime: run_log.jsonl + outer loop bridge files
│
├── tasks/research_tasks.json # Bootstrap evaluation dataset
└── program.md                # Outer-loop coding agent instructions
```
