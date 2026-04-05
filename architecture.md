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

## Continuous Self-Improvement Model

The system improves through three concurrent modes:

```
┌───────────────────────────────────────────────────────────────┐
│  OUTER LOOP (across many sessions)                             │
│  Coding agent reads evolution_state/ → makes structural changes│
│  Triggered when inner loop plateaus                            │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  BACKGROUND DAEMON (continuous, async)                    │  │
│  │  Watches run_log.jsonl → when 3+ runs accumulate:        │  │
│  │  prompt optimization + skill extraction + memory compress │  │
│  │  src/evolution/daemon.py                                  │  │
│  │                                                          │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │  HARNESS (per-step, within each run)               │  │  │
│  │  │  Middleware catches mistakes in real-time            │  │  │
│  │  │  evoagent/harness/middleware.py                      │  │  │
│  │  │                                                    │  │  │
│  │  │  SelfVerification → catches error-only outputs      │  │  │
│  │  │  ContextAssembly  → injects skills + task type      │  │  │
│  │  │  LoopDetection    → breaks repetitive search loops  │  │  │
│  │  │  TraceCapture     → records for offline analysis    │  │  │
│  │  │                                                    │  │  │
│  │  │  ┌──────────────────────────────────────────────┐  │  │  │
│  │  │  │  AGENT (per-task, seconds)                   │  │  │  │
│  │  │  │  Think → Search → Synthesize → Write report  │  │  │  │
│  │  │  │  Post-run: grade → score → reflect → log     │  │  │  │
│  │  │  └──────────────────────────────────────────────┘  │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                │
│  SLEEP-TIME COMPUTE (between sessions)                         │
│  Cross-run trace analysis → meta-instructions → memory         │
│  evoagent/evolution/sleep_review.py                            │
└───────────────────────────────────────────────────────────────┘
```

### How each mode feeds back

| Mode | Trigger | What it improves | Timescale |
|------|---------|-----------------|-----------|
| Per-run learning | Every `python -m src run` | Prompt score, failure feedback, episodic/semantic memory | Seconds |
| Background daemon | 3+ unprocessed runs in log | Prompt version, skills, failure skills, memory compression | Minutes |
| Batch evolution | `python -m src evolve` | Full optimization with holdout generalization check | Minutes–hours |
| Sleep-time review | `python -m src sleep-review` | Meta-instructions, cross-run patterns | Between sessions |
| Outer loop | Coding agent on plateau | Tools, model, architecture, grading pipeline | Hours–days |

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
skills/manager.py ◄────────  orchestrator.py, daemon.py, CLI (uses SkillManager)
skills/extractor.py ◄──────  orchestrator.py, daemon.py (uses extract_skills_from_batch)
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

**`core/parsing.py`** — `parse_llm_json()` handles markdown code blocks, trailing text, graceful fallback.

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

### Layer 3: Evolution (needs langgraph)

**`evolution/analyzer.py`**:
- `analyze_trajectory(graders, task, output, metrics)` — runs any `list[Grader]`, returns `list[GraderResult]`
- `classify_trajectory(results)` — classification with **critical grader gate**:
  - `CRITICAL_GRADERS = {"task_completion"}` — if a critical grader fails, classification is capped at "partial" regardless of average score
  - This ensures the prompt optimizer always receives signal when the most important dimension fails

**`evolution/sleep_review.py`**:
- Reads execution traces + episodic memories
- LLM identifies cross-cutting patterns
- Generates meta-instructions stored as semantic memories
- These are prioritized first in `compress_context()`

---

## Research Agent Application (`src/`)

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
| `METAPROMPT_TEMPLATE` | Prompt optimization with dimension breakdown + trace digest |
| `TC_COMPLETENESS/EVIDENCE/ACCURACY_PROMPT` | Multi-judge task completion |
| `Q_STRUCTURE/DEPTH/RELEVANCE_PROMPT` | Multi-judge quality |
| `CLAIM_EXTRACTION/VERIFICATION_PROMPT` | Claim-level verification |

**`prompt_store.py`** — Versioned prompt persistence (`prompts/v{NNNN}.json`):
- Parent-child lineage tracking
- Incremental score averaging (every single run contributes)
- Feedback accumulation from failing dimensions
- `get_current_prompt()` returns the best-scoring version

**`subagents.py`** — Research (web search only) + Synthesis (no search, skills injected) sub-agents.

### Evolution Pipeline (`src/evolution/`)

**`orchestrator.py`** — 11-node LangGraph pipeline:

```
START → run_batch → fetch_traces → analyze → reflect → compress_memories
    → extract_skills → create_failure_skills → optimize_prompt
    → holdout_check → persist_state → aggregate_metrics → [continue/END]
```

Key features:
- **Task sampling**: `_sample_tasks()` selects `batch_size` tasks per cycle with rotation (seeded by cycle number for reproducibility)
- **Train/holdout split**: `_split_tasks()` reserves ~30% of tasks for holdout generalization check
- `holdout_check` node runs after prompt optimization, scores independently (not fed back)
- Plateau detection: < 5% improvement for 2 consecutive cycles signals the outer loop

**`analyzer.py`** — 4-axis grading via LangGraph parallel fan-out:

| Grader | Implementation | Type | PASS threshold |
|--------|---------------|------|----------------|
| Task Completion | `MultiJudgeGrader` with 3 judge prompts | LLM | 0.75 |
| Efficiency | `EfficiencyGrader` from evoagent | Rule-based | 0.50 |
| Quality | `MultiJudgeGrader` with 3 judge prompts | LLM | 0.75 |
| Claim Verification | `grade_claims()` + optional web spot-check | LLM + Search | 0.60 |

**Critical grader gate**: if `task_completion` fails, the trajectory classification is capped at "partial" regardless of average score. This prevents the prompt optimizer from being starved of signal when other dimensions mask task_completion failures.

**`prompt_optimizer.py`** — Dimension-aware prompt rewriting:
- **Per-dimension breakdown**: the metaprompt shows the optimizer which grading dimensions are BOTTLENECK vs OK
- **Autonomy-safe retries**: when generated prompts contain forbidden patterns (e.g., "ask the user"), retries include specific violation feedback; as a last resort, `_strip_autonomy_violations()` removes offending lines
- **Explicit banned phrases**: the metaprompt lists exact forbidden strings rather than abstract rules
- Pairwise comparison: runs failed tasks with old and new prompt, LLM picks winner

**`daemon.py`** — Background evolution daemon:
- Polls `run_log.jsonl` at configurable intervals (default: 60s)
- When `min_runs` (default: 3) unprocessed entries accumulate, triggers lightweight evolution
- Never re-runs tasks — works entirely on grading results from real user interactions
- Runs: prompt optimization, skill extraction (success + failure), memory compression
- Graceful shutdown on SIGINT/SIGTERM

**`run_log.py`** — Append-only JSONL run log:
- Thread-safe appends (file-level lock)
- Each entry: run_id, task, output, classification, grading results, prompt version
- `mark_processed()` tracks which entries the daemon has consumed
- Written to by every `python -m src run`, read by the daemon

**`evolution_state_bridge.py`** — Persists state for outer-loop coding agent:

| File | Purpose |
|------|---------|
| `failures.md` | Deduplicated failure patterns with grader context |
| `hypotheses.md` | Cycle history + score progression |
| `skills_summary.md` | All skills (success + defensive) |
| `plateau_report.md` | Handoff signal when inner loop saturates |
| `run_log.jsonl` | Structured run results for daemon consumption |

**`state.py`** — LangGraph state schemas:
- `OrchestratorState`: includes `tasks`, `holdout_tasks`, `batch_size` for sampling/holdout
- `AnalyzerState`: 4 grader result slots + classification
- `EvolutionMetrics`: cycle summary with skills, prompt version, memories

### Research-Specific Graders (`src/evolution/graders/`)

| Grader | Function | Uses |
|--------|----------|------|
| `claim_verification.py` | `grade_claims()` + `_extract_claims()` | `parse_llm_json` from evoagent |
| `fact_checker.py` | `spot_check_claims()` (web search) | `parse_llm_json` from evoagent |

### Single Run Learning (`src/cli/commands.py`)

Every `cmd_run()` invocation:
1. Runs the agent on the task
2. Grades on 4 dimensions via `analyze_trajectory()`
3. Updates the current prompt version's score (incremental average)
4. Appends per-dimension failure feedback to the prompt version
5. Appends structured result to `run_log.jsonl` (for daemon)
6. Runs LLM reflection, stores episodic + semantic memories

### Other Modules

- **`memory/reflection.py`** — Post-run LLM introspection using `FileMemoryStore`
- **`tools/search.py`** — Tavily (primary) + DuckDuckGo (fallback) web search
- **`tracing/fetcher.py`** — LangSmith trace fetching using `TrajectoryRecord`
- **`config/settings.py`** — App-level Pydantic Settings with env var loading
- **`cli/commands.py`** — CLI: `run`, `evolve`, `evolve-daemon`, `prompts`, `skills`, `memory`, `state`, `sleep-review`

---

## Data Flow: Single Run → Background Evolution

```
User: python -m src run "Research fusion energy"
         │
         ▼
┌──────────────────────────────┐
│  Agent runs (with middleware) │
│  Output: research report     │
└──────────────┬───────────────┘
               │
         ┌─────┴──────┐
         ▼            ▼
┌──────────────┐ ┌──────────────────────┐
│  Grade (4x)  │ │  Print to user       │
│  TC, Eff,    │ └──────────────────────┘
│  Quality,    │
│  Claims      │
└──────┬───────┘
       │
  ┌────┴────┬──────────┬──────────────┐
  ▼         ▼          ▼              ▼
Score     Feedback   Reflect        Append to
prompt    on fails   → memory       run_log.jsonl
version   → prompt                       │
          version                        │ (daemon polls)
                                         ▼
                               ┌──────────────────┐
                               │ Background Daemon │
                               │ (when 3+ pending) │
                               └────────┬─────────┘
                                        │
                          ┌─────────────┼──────────────┐
                          ▼             ▼              ▼
                     Optimize       Extract         Compress
                     prompt         skills          memory
                          │             │
                          ▼             ▼
                   prompts/v*     skills/SKILL.md
                          │
                          ▼
                 Next run picks up improved prompt
```

## Data Flow: Batch Evolution Cycle

```
                 tasks/research_tasks.json
                         │
                    _split_tasks()
                    ┌────┴────┐
                    ▼         ▼
              training(5)  holdout(2)
                    │
              _sample_tasks(batch_size=3, cycle=N)
                    │
                    ▼
    ┌───────────────────────────────┐
    │         run_batch (3 tasks)   │
    │  Rotated sample each cycle   │
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  analyze (4 graders, parallel)│
    │  Critical grader gate:        │
    │  TC fail → capped at PARTIAL │
    └───────────────┬───────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
  ┌──────────────┐    ┌──────────────────────┐
  │   reflect    │    │  extract_skills      │
  │  → memory    │    │  success + failure   │
  └──────┬───────┘    └──────────┬───────────┘
         │                       │
         ▼                       │
  ┌──────────────┐               │
  │ compress_mem │               │
  └──────┬───────┘               │
         └───────────┬───────────┘
                     ▼
    ┌───────────────────────────────┐
    │     optimize_prompt            │
    │  Per-dimension breakdown:      │
    │  "task_completion: BOTTLENECK" │
    │  Violation feedback on retry   │
    │  Line stripping as last resort │
    └───────────────┬───────────────┘
                    ▼
    ┌───────────────────────────────┐
    │     holdout_check (2 tasks)   │
    │  Scores logged, not fed back  │
    │  Measures generalization      │
    └───────────────┬───────────────┘
                    ▼
    ┌───────────────────────────────┐
    │  persist_state + aggregate    │
    │  Plateau → signal outer loop  │
    └───────────────────────────────┘
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
| `CRITICAL_GRADERS` | `{"task_completion"}` | Both `analyzer.py` files | Graders that cap classification at partial when failing |
| `MIN_PASS_COUNT` | 2 | `evoagent/evolution/analyzer.py` | Graders that must pass for success |
| `MIN_AVERAGE_SCORE` | 0.60 | `evoagent/evolution/analyzer.py` | Min average for success classification |
| `MIN_IMPROVEMENT` | 0.05 | `src/evolution/orchestrator.py` | Plateau detection (5%) |
| `PLATEAU_CYCLES` | 2 | `src/evolution/orchestrator.py` | Consecutive low cycles to trigger plateau |
| `HOLDOUT_FRACTION` | 0.30 | `src/evolution/orchestrator.py` | Fraction of tasks reserved for holdout |
| `batch_size` | 3 | `src/config/settings.py` | Tasks sampled per evolution cycle |
| `memory_token_budget` | 4000 | `src/config/settings.py` | Max tokens for memory context |
| `compression_similarity_threshold` | 0.7 | `src/config/settings.py` | Jaccard threshold for semantic dedup |
| `MAX_SELF_CHECK_RETRIES` | 2 | `evoagent/harness/middleware.py` | Self-verification retry limit |
| `MAX_SIMILAR_QUERIES` | 3 | `evoagent/harness/middleware.py` | Loop detection trigger |
| `MAX_TOTAL_SEARCHES` | 12 | `evoagent/harness/middleware.py` | Total search cap before warning |
| `MAX_AUTONOMY_RETRIES` | 2 | `src/evolution/prompt_optimizer.py` | Prompt drift retry limit |
| `DEFAULT_POLL_INTERVAL` | 60 | `src/evolution/daemon.py` | Daemon polling interval (seconds) |
| `DEFAULT_MIN_RUNS` | 3 | `src/evolution/daemon.py` | Runs before daemon triggers evolution |

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
│   ├── evolution/            # Analyzer (with critical grader gate), sleep review, state
│   └── tracing/              # TrajectoryRecord, ToolCall
│
├── src/                      # Research agent application
│   ├── agent/                # deep_agent, prompts, prompt_store (with incremental scoring), subagents
│   ├── evolution/            # orchestrator (sampling + holdout), analyzer, prompt_optimizer
│   │   │                     #   (dimension-aware + violation feedback), daemon, run_log, graders/
│   ├── memory/               # reflection.py
│   ├── tools/                # search.py (Tavily + DuckDuckGo)
│   ├── tracing/              # fetcher.py (LangSmith)
│   ├── config/               # settings.py
│   └── cli/                  # commands.py (run, evolve, evolve-daemon, etc.)
│
├── memory/                   # Runtime: episodic + semantic memories
├── skills/                   # Runtime: learned SKILL.md files
├── prompts/                  # Runtime: versioned prompts with scores + feedback
├── traces/                   # Runtime: execution traces (JSON)
├── evolution_state/          # Runtime: inner→outer loop bridge + run_log.jsonl
│
├── tasks/research_tasks.json # Evaluation dataset
├── program.md                # Outer-loop coding agent instructions
└── results.tsv               # Outer-loop experiment log
```
