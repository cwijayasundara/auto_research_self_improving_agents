# Architecture: Auto-Research Self-Improving Agents

This document describes the architecture of the self-improving research agent system, built on the **`evoagent`** reusable library.

## Research Foundations

This project synthesizes techniques from five sources:

| Source | Contribution | Layer |
|--------|-------------|-------|
| [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) | Outer-loop: propose → run → evaluate → accept/reject | Outer Loop |
| [LangChain Harness Engineering](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) | Middleware: self-verification, loop detection, context assembly | Harness |
| [Stanford Meta-Harness](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact) | Environment bootstrapping, trace-driven optimization | Harness |
| [Letta Continual Learning](https://www.letta.com/blog/continual-learning) | Token-space learning, sleep-time compute, memory refinement | Sleep-Time |
| [Harrison Chase](https://x.com/hwchase17/article/2040467997022884194) | Three-layer learning model (model, harness, context) | All |

## 4-Layer Self-Improvement Model

Each layer operates at a different timescale:

```
┌───────────────────────────────────────────────────────────┐
│  OUTER LOOP (across many runs)                            │
│  Karpathy-style evolution: propose prompt → eval → keep?  │
│  src/evolution/orchestrator.py + prompt_optimizer.py       │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  HARNESS (per-step, within a run)                   │  │
│  │  Middleware catches mistakes in real-time            │  │
│  │  evoagent/harness/middleware.py                      │  │
│  │                                                     │  │
│  │  SelfVerification → catches error-only outputs      │  │
│  │  ContextAssembly  → injects skills + task type      │  │
│  │  LoopDetection    → breaks repetitive search loops  │  │
│  │  TraceCapture     → records for offline analysis    │  │
│  │                                                     │  │
│  │  ┌───────────────────────────────────────────────┐  │  │
│  │  │  INNER LOOP (per-task, seconds)               │  │  │
│  │  │  The agent reasoning, searching, writing      │  │  │
│  │  │  src/agent/deep_agent.py + tools + subagents  │  │  │
│  │  └───────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
│  SLEEP-TIME COMPUTE (between runs)                        │
│  Cross-run trace analysis → meta-instructions → memory    │
│  evoagent/evolution/sleep_review.py                       │
└───────────────────────────────────────────────────────────┘
```

---

## Two-Package Architecture

The codebase is split into a **reusable library** (`evoagent/`) and a **reference application** (`src/`):

```
evoagent/                    src/
(generic patterns)           (research-specific application)
                             
core/types.py ◄──────────── evolution/state.py (imports GraderResult)
core/protocols.py ◄──────── agent/deep_agent.py (implements AgentFactory pattern)
core/parsing.py ◄──────────  evolution/graders/*.py (uses parse_llm_json)
memory/store.py ◄──────────  cli/commands.py, orchestrator.py (uses FileMemoryStore)
memory/compression.py ◄────  orchestrator.py (uses deduplicate_semantic)
skills/manager.py ◄────────  orchestrator.py, CLI (uses SkillManager)
skills/extractor.py ◄──────  orchestrator.py (uses extract_skills_from_batch)
harness/middleware.py ◄────  agent/deep_agent.py (wires middleware stack)
graders/efficiency.py ◄────  evolution/analyzer.py (uses EfficiencyGrader)
graders/multi_judge.py ◄───  evolution/analyzer.py (uses MultiJudgeGrader)
evolution/analyzer.py ◄────  evolution/analyzer.py (uses classify_trajectory)
evolution/sleep_review.py ◄  cli/commands.py (sleep-review command)
tracing/trajectory.py ◄────  evolution/state.py, tracing/fetcher.py
```

---

## EvoAgent Library (`evoagent/`)

### Layer 0: Core (pydantic only)

**`core/types.py`** — Shared data types as plain dataclasses:
- `GraderResult`: name, score (0.0-1.0), passed, reasoning
- `TaskResult`: task, output, tool_calls, duration_seconds, status
- `TrajectoryMetrics`: total_tokens, total_steps, latency_seconds, tool_call_count
- `EvolutionCycleReport`: cycle, avg_score, classification_counts, prompt_version, skills_created

**`core/protocols.py`** — ABCs that define pluggable contracts:
- `Grader`: `.grade(task, output, **kwargs) -> GraderResult`
- `AgentFactory`: `.create(system_prompt, middleware) -> agent`, `.run(agent, task) -> TaskResult`
- `MemoryBackend`: `.store()`, `.retrieve()`, `.list_all()`, `.search()`, `.count()`, `.delete()`
- `SkillStore`: `.discover()`, `.load()`, `.create()`
- `PromptStore`: `.get_current()`, `.save()`, `.update_score()`

**`core/config.py`** — `EvoAgentConfig` (Pydantic Settings) with defaults for evolution, memory, and directory paths.

**`core/parsing.py`** — `parse_llm_json()` handles markdown code blocks, trailing text, graceful fallback. Replaces 4+ duplicate implementations across the old codebase.

### Layer 1: Memory & Skills (core only)

**`memory/store.py`** — `FileMemoryStore` implementing `MemoryBackend`:
- Two namespaces: `episodic` (run experiences) and `semantic` (facts/patterns)
- JSON file per memory item: `memory/{namespace}/{key}.json`
- Keyword search with relevance scoring

**`memory/compression.py`**:
- `compress_context()` — token-budgeted assembly with priority: meta-instructions > semantic > episodic
- `deduplicate_semantic()` — Jaccard word similarity, removes near-duplicates above threshold

**`skills/manager.py`** — `SkillManager` implementing `SkillStore`:
- SKILL.md files with YAML frontmatter (Anthropic skill-creator format)
- Progressive disclosure: name + description at discovery, full body on-demand
- Create, validate, discover operations

**`skills/extractor.py`** — Unified success + failure skill extraction:
- `mode="success"`: extract from runs scoring >= 0.70
- `mode="failure"`: extract defensive "antibody" skills from low-scoring runs
- Duplicate detection by word overlap

### Layer 2: Harness & Graders (needs langchain-core)

**`harness/middleware.py`** — Four middleware classes inheriting from `AgentMiddleware`:

| Middleware | Hook | Purpose |
|-----------|------|---------|
| `SelfVerificationMiddleware` | `after_model` | Checks output for errors/missing sections, injects revision prompt (max 2 retries) |
| `ContextAssemblyMiddleware` | `before_model` | Enriches first call with available skills and task classification |
| `LoopDetectionMiddleware` | `wrap_tool_call` | Tracks search queries, warns on >3 similar or >12 total |
| `TraceCaptureMiddleware` | `before/after_agent`, `wrap_*` | Records all calls to disk for offline analysis |

All parameterized via constructor (required_sections, max_retries, max_similar, etc.).

**`graders/efficiency.py`** — `EfficiencyGrader` (rule-based, no LLM):

| Metric | Ideal (1.0) | Acceptable (0.5) | Weight |
|--------|-------------|-------------------|--------|
| Tokens | < 10,000 | < 50,000 | 50% |
| Steps | < 5 | < 15 | 30% |
| Latency | < 30s | < 120s | 20% |

**`graders/multi_judge.py`** — `MultiJudgeGrader` (parallel LLM judges):
- Runs 3 perspective judges via `ThreadPoolExecutor`
- Aggregates via median
- Flags `[low_agreement]` when spread > 0.3
- Falls back to single judge if < 2 succeed
- Constructor accepts custom `judge_prompts` and `fallback_prompt`

### Layer 3: Evolution (needs langgraph)

**`evolution/analyzer.py`**:
- `analyze_trajectory(graders, task, output, metrics)` — runs any `list[Grader]`, returns `list[GraderResult]`
- `classify_trajectory(results)` — successful (avg >= 0.75, >= 2 pass), partial (avg >= 0.50), failed

**`evolution/prompt_optimizer.py`**:
- Analyzes failures, generates improved prompt via metaprompt
- Autonomy validation: blocks prompts that ask for user input (regex patterns)
- Up to 2 retries on autonomy violations

**`evolution/sleep_review.py`**:
- Reads execution traces + episodic memories
- LLM identifies cross-cutting patterns
- Generates meta-instructions stored as semantic memories
- These are prioritized first in `compress_context()`

**`evolution/state.py`** — `persist_evolution_state()` writes cycle metrics and analyses to disk.

### Tracing

**`tracing/trajectory.py`** — `TrajectoryRecord` (Pydantic model) with flat fields: run_id, task, output, tool_calls, total_tokens, total_steps, latency_seconds, status.

---

## Research Agent Application (`src/`)

The `src/` directory is a reference application that uses evoagent for research task automation.

### Agent Core (`src/agent/`)

**`deep_agent.py`** — Agent factory:
- Creates LLM via `init_chat_model()` (20+ providers)
- Retrieves best prompt from `PromptStore`
- Builds memory context via `compress_context()` from evoagent (token budget: 4000)
- Wires evoagent middleware stack: SelfVerification → ContextAssembly → LoopDetection → TraceCapture
- Optionally attaches research + synthesis sub-agents
- Returns `deepagents` compiled agent

**`prompts.py`** — Research-specific prompt templates:

| Prompt | Purpose |
|--------|---------|
| `DEFAULT_SYSTEM_PROMPT` | Research workflow (break → search → synthesize) |
| `REFLECTION_PROMPT` | Post-run introspection |
| `SKILL_EXTRACTION_PROMPT` | Extract patterns from successes |
| `FAILURE_SKILL_PROMPT` | Create defensive skills from failures |
| `METAPROMPT_TEMPLATE` | Prompt optimization with trace digest |
| `TC_COMPLETENESS/EVIDENCE/ACCURACY_PROMPT` | Multi-judge task completion |
| `Q_STRUCTURE/DEPTH/RELEVANCE_PROMPT` | Multi-judge quality |
| `CLAIM_EXTRACTION/VERIFICATION_PROMPT` | Claim-level verification |

**`prompt_store.py`** — Versioned prompt persistence (`prompts/v{NNNN}.json`) with parent-child lineage and score tracking.

**`subagents.py`** — Research (web search only) + Synthesis (no search, skills injected) sub-agents.

### Evolution Pipeline (`src/evolution/`)

**`orchestrator.py`** — 10-node LangGraph pipeline:

```
START → run_batch → fetch_traces → analyze → reflect → compress_memories
    → extract_skills → create_failure_skills → optimize_prompt
    → persist_state → aggregate_metrics → [continue/END]
```

Uses evoagent components throughout:
- `FileMemoryStore` for memory
- `SkillManager` for skill CRUD
- `extract_skills_from_batch(mode="success")` and `extract_skills_from_batch(mode="failure")`
- `deduplicate_semantic()` for memory compression
- `EfficiencyGrader` and `MultiJudgeGrader` for evaluation (via analyzer)

**`analyzer.py`** — 4-axis grading via LangGraph parallel fan-out:

| Grader | Implementation | Type |
|--------|---------------|------|
| Task Completion | `MultiJudgeGrader` with 3 research-specific judge prompts | LLM |
| Efficiency | `EfficiencyGrader` from evoagent | Rule-based |
| Quality | `MultiJudgeGrader` with 3 research-specific judge prompts | LLM |
| Claim Verification | Research-specific `grade_claims()` + optional web spot-check | LLM + Search |

**`prompt_optimizer.py`** — Skill-aware prompt rewriting with:
- Trace digest from `TraceCaptureMiddleware` recordings
- Pairwise comparison: runs failed tasks with old and new prompt, LLM picks winner
- Autonomy validation (blocks "ask the user" drift)

**`evolution_state_bridge.py`** — Persists state for outer-loop coding agent:

| File | Purpose |
|------|---------|
| `failures.md` | Deduplicated failure patterns with grader context |
| `hypotheses.md` | Cycle history + score progression |
| `skills_summary.md` | All skills (success + defensive) |
| `plateau_report.md` | Handoff signal when inner loop saturates |

**`state.py`** — LangGraph state schemas (`AnalyzerState`, `OrchestratorState`, etc.). Imports `GraderResult` and `TrajectoryRecord` from evoagent.

### Research-Specific Graders (`src/evolution/graders/`)

| Grader | Function | Uses |
|--------|----------|------|
| `quality.py` | `grade_quality()` | `parse_llm_json` from evoagent |
| `task_completion.py` | `grade_task_completion()` | `parse_llm_json` from evoagent |
| `claim_verification.py` | `grade_claims()` + `_extract_claims()` | `parse_llm_json` from evoagent |
| `fact_checker.py` | `spot_check_claims()` (web search) | `parse_llm_json` from evoagent |

### Other Modules

- **`memory/reflection.py`** — Post-run LLM introspection using `FileMemoryStore`
- **`tools/search.py`** — Tavily (primary) + DuckDuckGo (fallback) web search
- **`tracing/fetcher.py`** — LangSmith trace fetching using `TrajectoryRecord`
- **`config/settings.py`** — App-level Pydantic Settings with env var loading
- **`cli/commands.py`** — CLI: `run`, `evolve`, `prompts`, `skills`, `memory`, `state`, `sleep-review`

---

## Data Flow: Complete Evolution Cycle

```
                     tasks/research_tasks.json
                                │
                                ▼
                ┌───────────────────────────────┐
                │         run_batch              │
                │  For each task:                │
                │  1. Load best prompt (PromptStore)
                │  2. Build memory context       │
                │     (evoagent compress_context) │
                │  3. Wire middleware stack       │
                │     (evoagent harness)          │
                │  4. Invoke agent               │
                │  5. Collect TrajectoryRecord    │
                └───────────────┬───────────────┘
                                │
                                ▼
                ┌───────────────────────────────┐
                │          analyze               │
                │  For each trajectory (parallel):│
                │  ├─ MultiJudgeGrader (TC)      │
                │  ├─ EfficiencyGrader           │
                │  ├─ MultiJudgeGrader (quality) │
                │  ├─ grade_claims + spot_check  │
                │  └─ classify_trajectory        │
                └───────────────┬───────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
      ┌──────────────────┐    ┌──────────────────────┐
      │     reflect      │    │  extract_skills      │
      │  LLM introspects │    │  (evoagent extractor) │
      │  → episodic mem  │    │  mode="success"       │
      │  → semantic mem  │    │  mode="failure"       │
      └────────┬─────────┘    └──────────┬───────────┘
               │                         │
               ▼                         │
      ┌──────────────────┐               │
      │ compress_memories│               │
      │  (evoagent dedup)│               │
      └────────┬─────────┘               │
               └───────────┬─────────────┘
                           ▼
                ┌───────────────────────────┐
                │     optimize_prompt        │
                │  Trace digest + failures   │
                │  + pairwise comparison     │
                │  + autonomy validation     │
                │  → new prompt version      │
                └───────────┬───────────────┘
                           ▼
                ┌───────────────────────────┐
                │      persist_state         │
                │  → evolution_state/        │
                └───────────┬───────────────┘
                           ▼
                ┌───────────────────────────┐
                │    aggregate_metrics       │
                │  plateau check → continue? │
                └───────────────────────────┘
```

### Inner → Outer Loop Handoff

When the inner loop plateaus (< 5% improvement for 2 consecutive cycles):

```
Inner Loop                          Outer Loop (Coding Agent)
    │                                       │
    │  writes evolution_state/              │
    │  ├── failures.md ──────────────────→  │ "These failures need structural changes"
    │  ├── hypotheses.md ────────────────→  │ "Here's what was tried and worked"
    │  ├── skills_summary.md ────────────→  │ "Agent knows these patterns"
    │  └── plateau_report.md ────────────→  │ "Inner loop saturated, your turn"
    │                                       │
    │                                       │  1. Makes ONE structural change
    │                                       │  2. Commits to git
    │  ◀──── re-runs inner loop ─────────── │  3. Re-evaluates
    │                                       │  4. KEEP or DISCARD
```

---

## Critical Thresholds

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `SUCCESS_THRESHOLD` | 0.70 | `evoagent/skills/extractor.py` | Min score to extract success skills |
| `FAILURE_THRESHOLD` | 0.50 | `evoagent/skills/extractor.py` | Max score for failure skill extraction |
| `MAX_SKILLS_PER_CYCLE` | 3 | `evoagent/skills/extractor.py` | Prevent skill explosion |
| `SUCCESSFUL_THRESHOLD` | 0.75 | `evoagent/evolution/analyzer.py` | Classification as successful |
| `PARTIAL_THRESHOLD` | 0.50 | `evoagent/evolution/analyzer.py` | Classification as partial |
| `MIN_PASS_COUNT` | 2 | `evoagent/evolution/analyzer.py` | Graders that must pass for success |
| `MIN_AVERAGE_SCORE` | 0.60 | `evoagent/evolution/analyzer.py` | Min average for success classification |
| `MIN_IMPROVEMENT` | 0.05 | `src/evolution/orchestrator.py` | Plateau detection (5%) |
| `PLATEAU_CYCLES` | 2 | `src/evolution/orchestrator.py` | Consecutive low cycles to trigger plateau |
| `memory_token_budget` | 4000 | `src/config/settings.py` | Max tokens for memory context |
| `compression_similarity_threshold` | 0.7 | `src/config/settings.py` | Jaccard threshold for semantic dedup |
| `MAX_SELF_CHECK_RETRIES` | 2 | `evoagent/harness/middleware.py` | Self-verification retry limit |
| `MAX_SIMILAR_QUERIES` | 3 | `evoagent/harness/middleware.py` | Loop detection trigger |
| `MAX_TOTAL_SEARCHES` | 12 | `evoagent/harness/middleware.py` | Total search cap before warning |
| `MAX_AUTONOMY_RETRIES` | 2 | `evoagent/evolution/prompt_optimizer.py` | Prompt drift retry limit |

---

## Technology Stack

| Component | Technology | Package |
|-----------|-----------|---------|
| Core types & protocols | `pydantic`, `pydantic-settings` | `evoagent` (core) |
| Memory & skills | File-backed JSON, SKILL.md | `evoagent` (core) |
| Middleware | `AgentMiddleware` | `evoagent[harness]` |
| Graders | `ThreadPoolExecutor`, LLM-as-judge | `evoagent[graders]` |
| Evolution loop | `langgraph` StateGraph | `evoagent[evolution]` |
| Agent framework | `deepagents` | `src/` app |
| LLM abstraction | `langchain` + `init_chat_model()` | `src/` app |
| Web search | `langchain-tavily` + `ddgs` | `src/` app |
| Tracing | `langsmith` | `src/` app |

---

## Persistence Layout

```
auto_research_self_improving_agents/
├── evoagent/                 # Reusable library (pip install -e "./evoagent[all]")
│   ├── core/                 # Types, protocols, config, parsing
│   ├── memory/               # FileMemoryStore, compression
│   ├── skills/               # SkillManager, extractor
│   ├── harness/              # 4 middleware classes + builder
│   ├── graders/              # EfficiencyGrader, MultiJudgeGrader
│   ├── evolution/            # Analyzer, prompt optimizer, sleep review, state
│   └── tracing/              # TrajectoryRecord, ToolCall
│
├── src/                      # Research agent application
│   ├── agent/                # deep_agent, prompts, prompt_store, subagents
│   ├── evolution/            # orchestrator, analyzer, prompt_optimizer, graders/
│   ├── memory/               # reflection.py
│   ├── tools/                # search.py (Tavily + DuckDuckGo)
│   ├── tracing/              # fetcher.py (LangSmith)
│   ├── config/               # settings.py
│   └── cli/                  # commands.py
│
├── memory/                   # Runtime: episodic + semantic memories
├── skills/                   # Runtime: learned SKILL.md files
├── prompts/                  # Runtime: versioned prompts (v0001.json, ...)
├── traces/                   # Runtime: execution traces (JSON)
├── evolution_state/          # Runtime: inner→outer loop bridge files
│
├── tasks/research_tasks.json # Evaluation dataset
├── program.md                # Outer-loop coding agent instructions
└── results.tsv               # Outer-loop experiment log
```
