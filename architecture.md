# Architecture: Auto-Research Self-Improving Agents

This document describes the detailed design of this repository, which synthesizes two parent approaches into a unified two-speed self-improving research agent system.

## Parent Approaches

### 1. Karpathy's Autoresearch (Outer Loop)

**Source**: [autoresearch-agents](https://github.com/cwijayasundara/autoresearch-agents)

The autoresearch approach treats agent improvement as a **greedy hill-climbing experiment loop** driven by an external coding agent (Claude Code, Cursor, Codex). The coding agent:

1. Reads `program.md` for instructions and constraints
2. Hypothesizes an improvement (prompt tuning, tool changes, model swaps, architecture refactors)
3. Edits `agent.py` — the **only mutable file** during experimentation
4. Commits the change to git
5. Runs `python run_eval.py > eval.log 2>&1` against a fixed dataset (20 examples)
6. Parses scores: `grep "^overall_score:" eval.log`
7. **KEEP** if score improved, **DISCARD** (`git reset --hard`) if not
8. Logs result to `results.tsv` (commit, scores, status, description)
9. Loops indefinitely until interrupted

**Key properties**:
- Immutable evaluation pipeline — only `agent.py` changes
- Three evaluators: correctness (LLM-as-judge), helpfulness (LLM-as-judge), tool_usage (code-based)
- Greedy search with simplicity bias (simpler code preferred at equal scores)
- Full LangSmith traceability per experiment
- Git commits as experiment snapshots enabling reproducibility

**Limitation**: No automatic learning — the coding agent must reason about *what* to change. No memory, no skill accumulation, no prompt optimization. All intelligence lives in the coding agent's reasoning.

### 2. Self-Evolving Deep Agents (Inner Loop)

**Source**: [self_evolving_deep_agents](https://github.com/cwijayasundara/self_evolving_deep_agents)

The self-evolving approach implements a **closed-loop evolution engine** where the agent improves itself automatically through prompt optimization, skill extraction, and reflective memory:

1. **Run batch** — execute agent on all tasks with memory-augmented prompts
2. **Fetch traces** — pull LangSmith traces for observability
3. **Analyze** — grade each trajectory via 3-axis evaluation (task_completion, efficiency, quality)
4. **Reflect** — LLM generates introspection, stores episodic + semantic memories
5. **Compress memories** — deduplicate semantic, consolidate old episodic
6. **Extract skills** — from successful runs (score >= 0.70)
7. **Optimize prompt** — failure-aware metaprompt rewriting
8. **Aggregate metrics** — plateau detection (< 5% improvement x 2 consecutive cycles)

**Key properties**:
- LangGraph StateGraph orchestration with conditional routing
- Dual memory system (episodic experiences + semantic facts/patterns)
- Token-budgeted memory injection into prompts
- Versioned prompt store with parent-child lineage
- SKILL.md format with progressive disclosure (name/description at startup, full body on-demand)
- Multi-provider LLM support via `init_chat_model()` (20+ providers)

**Limitation**: Cannot make structural changes — only optimizes prompts, memory, and skills. When the inner loop plateaus, there is no mechanism to add new tools, swap architectures, or change the agent's code.

---

## This Repository: Two-Speed Synthesis

This repo combines both approaches into a two-speed system where the **inner loop** handles automatic optimization and the **outer loop** handles structural changes guided by inner-loop telemetry.

```
┌──────────────────────────────────────────────────────────────────┐
│                   OUTER LOOP: Coding Agent                       │
│              (Claude Code / Cursor / Codex)                      │
│                                                                  │
│  Reads: program.md + evolution_state/ + results.tsv              │
│  Writes: agent code, tools, architecture                         │
│  Strategy: Greedy hill-climbing on structural changes             │
│                                                                  │
│  Decision Protocol:                                              │
│  1. Read evolution_state/*.md                                    │
│  2. If failures are prompt-related → skip (inner loop handles)   │
│  3. If capability gaps → add new tools                           │
│  4. If architectural → refactor agent structure                  │
│  5. If model limitations → swap model                            │
│  6. If all scores high → expand evaluation dataset               │
│  7. Make ONE change, commit, re-run inner loop                   │
├──────────────────────────────────────────────────────────────────┤
│                   INNER LOOP: Evolution Engine                    │
│                                                                  │
│  ┌──────────┐   ┌─────────┐   ┌───────────────────┐             │
│  │ Run Batch│ → │ Grade   │ → │ Reflect & Learn   │             │
│  │ (agent)  │   │ (3 axes)│   │ (memory + skills) │             │
│  └──────────┘   └─────────┘   └───────────────────┘             │
│        │                              │                          │
│        │     ┌─────────────────┐      │                          │
│        └───→ │ Failure Skills  │ ←────┘                          │
│              │ (antibodies)    │                                  │
│              └────────┬────────┘                                  │
│                       ↓                                          │
│              ┌─────────────────┐                                  │
│              │ Prompt Optimize │                                  │
│              │ (skill-aware)   │                                  │
│              └────────┬────────┘                                  │
│                       ↓                                          │
│              ┌─────────────────┐                                  │
│              │ Persist State   │ → evolution_state/               │
│              └─────────────────┘                                  │
├──────────────────────────────────────────────────────────────────┤
│                   EVAL: 3-Axis Graders                           │
│  task_completion (LLM) + efficiency (rule) + quality (LLM)       │
└──────────────────────────────────────────────────────────────────┘
```

### Key Innovation: Failure-Driven Skill Creation

Neither parent approach learns defensive patterns from failures. This repo adds **failure skill creation** — when the agent fails repeatedly at a pattern, it creates "antibody" SKILL.md files that are injected into future runs to prevent recurrence.

---

## Module Architecture

### 1. Agent Core (`src/agent/`)

Responsible for constructing and configuring the research agent.

#### `deep_agent.py` — Agent Factory

```python
create_agent(settings, prompt_store, memory_store, task) -> CompiledStateGraph
```

- Creates LLM via `init_chat_model()` with provider abstraction (OpenAI, Anthropic, Ollama, Google, Groq, etc.)
- Retrieves current best prompt from `PromptStore`
- Builds memory context via `build_memory_context()` — searches episodic + semantic stores by task keywords, respects token budget (default 1000 tokens)
- Injects memory context into prompt via `{memory_context}` placeholder
- Discovers skills from `skills/` directory (progressive disclosure)
- Optionally attaches research + synthesis sub-agents
- Returns LangGraph `CompiledStateGraph` ready for invocation

**Memory injection flow**:
```
Task keywords
    ↓
Search semantic store (facts, patterns) → ranked by relevance
    ↓
Search episodic store (past runs) → ranked by relevance
    ↓
Build context within token budget (1000 tokens, ~4 chars/token)
    ↓
Format as markdown: "## Relevant Past Experience\n### Learned Facts..."
    ↓
Inject via prompt.format(memory_context=context)
```

#### `prompts.py` — Prompt Templates

All prompt templates used across the system:

| Prompt | Used By | Purpose |
|--------|---------|---------|
| `DEFAULT_SYSTEM_PROMPT` | Agent | Research workflow (break → search → synthesize) |
| `REFLECTION_PROMPT` | Reflection engine | Post-run introspection (strategy, worked_well, improvements, facts, patterns) |
| `SKILL_EXTRACTION_PROMPT` | Skill extractor | Extract reusable patterns from successful runs |
| `FAILURE_SKILL_PROMPT` | Failure skill creator | Create defensive skills from failure patterns |
| `METAPROMPT_TEMPLATE` | Prompt optimizer | Skill-aware prompt rewriting |
| `TASK_COMPLETION_PROMPT` | Task completion grader | LLM-as-judge for task completion |
| `QUALITY_PROMPT` | Quality grader | LLM-as-judge for output quality |

#### `prompt_store.py` — Versioned Prompt Persistence

```python
class PromptVersion:
    version: int
    prompt: str
    score: float | None
    parent_version: int | None
    feedback_summary: str
    timestamp: str
```

- Stored as `prompts/v{NNNN}.json`
- Parent-child lineage enables rollback
- Score tracking guides version selection (highest-scoring or latest if unscored)

#### `subagents.py` — Research/Synthesis Isolation

- `build_research_subagent()`: Web search only, raw data gathering
- `build_synthesis_subagent()`: No search tools, receives gathered data, skills injected

---

### 2. Evolution Engine (`src/evolution/`)

The core self-improvement loop, built as a 10-node LangGraph pipeline.

#### `orchestrator.py` — Main Evolution Loop

**10-Node Pipeline**:

```
START
  │
  ▼
┌─────────────┐
│  run_batch   │  Execute agent on all tasks (memory-augmented, skills-loaded)
└──────┬──────┘
       ▼
┌─────────────┐
│ fetch_traces │  Pull LangSmith traces (currently disabled; uses local trajectories)
└──────┬──────┘
       ▼
┌─────────────┐
│   analyze    │  Grade each trajectory via 3-axis evaluation
└──────┬──────┘
       ▼
┌─────────────┐
│   reflect    │  LLM generates introspection → episodic + semantic memories
└──────┬──────┘
       ▼
┌──────────────────┐
│ compress_memories │  Deduplicate semantic, consolidate old episodic
└──────┬───────────┘
       ▼
┌────────────────┐
│ extract_skills  │  From successful runs (score >= 0.70)
└──────┬─────────┘
       ▼
┌───────────────────────┐
│ create_failure_skills  │  From failure patterns (>= 2 failures per pattern)
└──────┬────────────────┘
       ▼
┌─────────────────┐
│ optimize_prompt  │  Skill-aware metaprompt → new prompt version
└──────┬──────────┘
       ▼
┌────────────────┐
│ persist_state   │  Write evolution_state/ for outer loop
└──────┬─────────┘
       ▼
┌───────────────────┐
│ aggregate_metrics  │  Compute cycle summary, plateau detection
└──────┬────────────┘
       ▼
   ┌────────┐
   │ route?  │──── improvement > 5% AND cycles < max → back to run_batch
   └────────┘──── otherwise → END
```

**Plateau detection**: If the last 2 consecutive cycles show < 5% improvement, the loop terminates and writes `plateau_report.md` to signal the outer-loop coding agent.

#### `analyzer.py` — 3-Axis Trajectory Analysis

LangGraph 4-node pipeline: `task_completion → efficiency → quality → classify`

**Classification logic**:

| Classification | Criteria |
|---------------|----------|
| **Successful** | >= 2 graders pass AND avg score >= 0.60 AND overall >= 0.75 |
| **Partial** | >= 2 graders pass AND avg >= 0.60 (but score < 0.75), OR avg >= 0.50 |
| **Failed** | Everything else |

#### `skill_extractor.py` — Success Skill Extraction

- **Threshold**: Score >= 0.70
- LLM analyzes successful trajectory → extracts reusable pattern
- Deduplication by word overlap (>= 2 shared words with existing skill → skip)
- Creates `skills/{skill-id}/SKILL.md` with YAML frontmatter

#### `failure_skill_creator.py` — Defensive Skill Creation (Key Innovation)

This is what distinguishes this repo from both parent approaches.

**Process**:
1. Group failed trajectories by worst-performing grader (pattern detection)
2. Only create skills for patterns with >= 2 failures (`MIN_FAILURES_FOR_SKILL`)
3. Cap at 3 per cycle (`MAX_FAILURE_SKILLS_PER_CYCLE`) to prevent skill explosion
4. LLM generates defensive SKILL.md via `FAILURE_SKILL_PROMPT`
5. Skills prefixed `avoid-` or `handle-` (e.g., `handle-task-completion-failures`)

**Why this matters**: The agent literally immunizes itself against recurring failure modes. If it repeatedly fails at citing sources, a `handle-citation-requirements` skill is created and automatically injected into future runs.

#### `prompt_optimizer.py` — Skill-Aware Prompt Optimization

1. Analyze failures from current cycle (aggregate grader reasoning, top 10 issues)
2. Discover all learned skills (both success and defensive)
3. Call LLM with metaprompt: current prompt + failures + available skills
4. Generate improved prompt (must preserve `{memory_context}` placeholder)
5. Persist as new version with parent link + feedback summary

**Difference from parent**: The parent self-evolving repo's optimizer is skill-unaware. This repo's optimizer explicitly references learned skills so the prompt can leverage them.

#### `evolution_state_bridge.py` — Inner → Outer Loop Bridge

Persists human-readable state for the outer-loop coding agent:

| File | Content | Purpose |
|------|---------|---------|
| `failures.md` | Deduplicated failure patterns with grader context | Tell coding agent what's broken |
| `hypotheses.md` | Cycle history, score progression, current assessment | Show learning trajectory |
| `skills_summary.md` | Success-derived + defensive skills (separated) | Show what agent has learned |
| `plateau_report.md` | Final metrics + what to try next | Trigger structural intervention |

#### `state.py` — LangGraph State Schemas

```python
GraderResult:    name, score, passed, reasoning
AnalysisResult:  run_id, task, classification, average_score, grader_results[], output, tool_calls
EvolutionMetrics: avg_score, skills_learned, failure_skills, prompt_version, memories, trajectories_analyzed
OrchestratorState: tasks, cycle, trajectories, analyses, metrics_history[]
```

---

### 3. Grading System (`src/evolution/graders/`)

Three independent graders provide multi-axis evaluation:

#### `task_completion.py` — LLM-as-Judge

- Evaluates: Did the agent address the core question? Is the response structured? Are claims cited? Is it accurate?
- Score: 0.0–1.0
- Pass threshold: >= 0.75

#### `efficiency.py` — Rule-Based

No LLM call. Scores resource usage against ideal/acceptable thresholds:

| Metric | Ideal (1.0) | Acceptable (0.5) | Weight |
|--------|-------------|-------------------|--------|
| Tokens | < 10,000 | < 50,000 | 50% |
| Steps | < 5 | < 15 | 30% |
| Latency | < 30s | < 120s | 20% |

- Pass threshold: >= 0.50

#### `quality.py` — LLM-as-Judge

- Evaluates 4 dimensions: accuracy, depth, clarity, relevance (each 0.0–1.0)
- Overall score: weighted average
- Pass threshold: >= 0.75

---

### 4. Memory System (`src/memory/`)

Dual-store memory with compression and token budgeting.

#### `store.py` — File-Backed Memory Store

- Two namespaces: `episodic` (run experiences) and `semantic` (facts/patterns)
- Storage: `memory/{namespace}/{key}.json`
- Operations: `store()`, `retrieve()`, `list_all()`, `search()` (keyword matching with relevance scoring), `count()`

#### `reflection.py` — Post-Run Introspection

After each agent run:
1. LLM analyzes trajectory via `REFLECTION_PROMPT`
2. Generates structured JSON: `{strategy, worked_well, improvements, facts[], patterns[]}`
3. Stores full reflection as **episodic memory** (keyed by run_id)
4. Extracts each fact and pattern as **semantic memory** (UUID-keyed)

#### `compression.py` — Memory Management

Prevents unbounded growth:
- **Semantic deduplication**: Jaccard similarity > 0.7 → remove duplicate
- **Episodic consolidation**: Keep 10 most recent; consolidate older into date-bucketed summaries
- **Token budgeting**: `compress_memory_context()` builds context within budget (default 1000 tokens, ~4 chars/token)

---

### 5. Skills Management (`src/skills/`)

#### `manager.py` — SKILL.md CRUD

**Format** (Anthropic skill-creator style):
```yaml
---
name: skill-name
description: What it does AND when to trigger (assertive, context-specific)
---
# Instructions
[Detailed markdown content]
```

**Progressive disclosure**:
- `discover_skills()`: Loads only `name` + `description` from all skills at startup
- `load_skill()`: Full content on-demand when matched to a task
- `create_skill()`: Creates `skills/{skill-id}/SKILL.md`
- `validate_skill()`: Checks frontmatter, description, body presence

---

### 6. Tools (`src/tools/`)

#### `search.py` — Web Search

- Tavily web search configured for advanced depth, 5 results per query
- API key from settings

---

### 7. Tracing (`src/tracing/`)

#### `fetcher.py` — LangSmith Integration

- Currently disabled (returns empty list) to avoid trace pollution from inner-loop LLM calls (optimizer, skill creator, graders)
- Falls back to local trajectory data with correct task/output pairs

#### `trajectory.py` — Pydantic Models

```python
ToolCall:           name, args, output
TrajectoryMetrics:  total_tokens, total_steps, latency_seconds, tool_call_count
Trajectory:         run_id, task, output, tool_calls, metrics, messages, status, feedback
```

---

### 8. Configuration (`src/config/`)

#### `settings.py` — Pydantic-Settings

Environment-driven configuration with validation:

| Category | Keys |
|----------|------|
| API Keys | `openai_api_key`, `tavily_api_key`, `ollama_api_key`, `langsmith_api_key` |
| Model | `model`, `model_provider`, `model_base_url` |
| Evolution | `max_evolution_cycles`, `batch_size`, `outer_loop_enabled`, `min_improvement_threshold` |
| Memory | `memory_token_budget` (1000), `compression_similarity_threshold` (0.7) |
| Paths | `skills_dir`, `prompts_dir`, `memory_dir`, `evolution_state_dir` |
| Agent | `use_subagents` |
| Tracing | `langchain_tracing_v2`, `langsmith_project` |

---

### 9. CLI (`src/cli/`)

#### `commands.py` — Command Interface

| Command | Purpose |
|---------|---------|
| `run <task>` | Execute agent on a single task |
| `evolve --tasks-file <path> --max-cycles <n>` | Run inner-loop evolution |
| `prompts` | Show prompt version history with scores |
| `skills` | List all learned skills (success + defensive) |
| `memory` | Browse episodic + semantic memory stores |
| `state` | Show `evolution_state/` contents for outer-loop debugging |

---

## Data Flow: Complete Cycle

### Single Evolution Cycle

```
                         tasks/research_tasks.json
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │         run_batch              │
                    │  For each task:                │
                    │  1. Load best prompt version   │
                    │  2. Search memory by keywords  │
                    │  3. Inject memory into prompt  │
                    │  4. Load matching skills       │
                    │  5. Invoke agent               │
                    │  6. Collect trajectory          │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │          analyze               │
                    │  For each trajectory:          │
                    │  ├─ task_completion (LLM)      │
                    │  ├─ efficiency (rules)         │
                    │  ├─ quality (LLM)              │
                    │  └─ classify: success/partial/ │
                    │     failed                     │
                    └───────────────┬───────────────┘
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
          ┌──────────────────┐    ┌──────────────────────┐
          │     reflect      │    │  extract_skills      │
          │  LLM introspects │    │  score >= 0.70       │
          │  → episodic mem  │    │  → SKILL.md          │
          │  → semantic mem  │    │                      │
          └────────┬─────────┘    └──────────┬───────────┘
                   │                         │
                   ▼                         ▼
          ┌──────────────────┐    ┌──────────────────────┐
          │ compress_memories│    │ create_failure_skills │
          │  dedup + consolidate  │  >= 2 failures/pattern│
          └────────┬─────────┘    │  → defensive SKILL.md│
                   │              └──────────┬───────────┘
                   └───────────┬─────────────┘
                               ▼
                    ┌───────────────────────────┐
                    │     optimize_prompt        │
                    │  Analyze failures          │
                    │  + reference learned skills│
                    │  → new prompt version      │
                    └───────────┬───────────────┘
                               ▼
                    ┌───────────────────────────┐
                    │      persist_state         │
                    │  Write evolution_state/    │
                    │  (failures, hypotheses,    │
                    │   skills, plateau report)  │
                    └───────────┬───────────────┘
                               ▼
                    ┌───────────────────────────┐
                    │    aggregate_metrics       │
                    │  avg_score, plateau check  │
                    │  → continue or stop        │
                    └───────────────────────────┘
```

### Inner → Outer Loop Handoff

When the inner loop plateaus (< 5% improvement for 2 consecutive cycles):

```
Inner Loop                          Outer Loop (Coding Agent)
    │                                       │
    │  writes evolution_state/              │
    │  ├── failures.md ──────────────────→  │ "These failures need tools/architecture"
    │  ├── hypotheses.md ────────────────→  │ "Here's what was tried and worked"
    │  ├── skills_summary.md ────────────→  │ "Agent knows these patterns"
    │  └── plateau_report.md ────────────→  │ "Inner loop saturated, your turn"
    │                                       │
    │                                       │  1. Reads state files
    │                                       │  2. Makes ONE structural change
    │                                       │  3. Commits to git
    │                                       │  4. Re-runs inner loop
    │  ◀──────── make evolve ──────────── │
    │                                       │
    │  (new cycle with structural change)   │  5. Compares results
    │  writes updated evolution_state/      │  6. KEEP or DISCARD (git reset)
    │  ────────────────────────────────→    │  7. Logs to results.tsv
    │                                       │  8. Loops
```

---

## Critical Thresholds & Constants

| Constant | Value | Location | Purpose |
|----------|-------|----------|---------|
| `SKILL_EXTRACTION_THRESHOLD` | 0.70 | `skill_extractor.py` | Min score to extract success skills |
| `MIN_FAILURES_FOR_SKILL` | 2 | `failure_skill_creator.py` | Min failures to create a defensive skill |
| `MAX_FAILURE_SKILLS_PER_CYCLE` | 3 | `failure_skill_creator.py` | Prevent skill explosion |
| `SUCCESSFUL_THRESHOLD` | 0.75 | `analyzer.py` | Classification as successful |
| `PARTIAL_THRESHOLD` | 0.50 | `analyzer.py` | Classification as partial |
| `MIN_PASS_COUNT` | 2 | `analyzer.py` | Graders that must pass for success |
| `MIN_AVERAGE_SCORE` | 0.60 | `analyzer.py` | Min average for success classification |
| `MIN_IMPROVEMENT` | 0.05 | `orchestrator.py` | Plateau detection threshold (5%) |
| `PLATEAU_CYCLES` | 2 | `orchestrator.py` | Consecutive low-improvement cycles to trigger plateau |
| `memory_token_budget` | 1000 | `settings.py` | Max tokens for memory context injection |
| `compression_similarity_threshold` | 0.7 | `settings.py` | Jaccard similarity for semantic dedup |

---

## Module Dependency Graph

```
CLI (commands.py)
├── create_agent (deep_agent.py)
│   ├── init_chat_model ← settings (model, provider)
│   ├── PromptStore (prompt_store.py)
│   ├── MemoryStore (store.py) → compression.py
│   ├── create_search_tool (search.py)
│   ├── discover_skills (manager.py)
│   └── Subagents (subagents.py) [optional]
│
├── run_evolution (orchestrator.py)
│   ├── create_agent (deep_agent.py)
│   ├── analyze_trajectory (analyzer.py)
│   │   ├── grade_task_completion (task_completion.py)
│   │   ├── grade_efficiency (efficiency.py)
│   │   └── grade_quality (quality.py)
│   ├── reflect_and_store (reflection.py) → MemoryStore
│   ├── compress_all (compression.py) → MemoryStore
│   ├── extract_skills_from_batch (skill_extractor.py) → manager.py
│   ├── create_failure_skills (failure_skill_creator.py) → manager.py
│   ├── optimize_prompt (prompt_optimizer.py) → manager.py
│   ├── persist_evolution_state (evolution_state_bridge.py) → manager.py
│   └── PromptStore, MemoryStore
│
├── PromptStore (prompt_store.py)
├── MemoryStore (store.py)
└── discover_skills (manager.py)
```

---

## Comparison: What Each Approach Contributes

| Capability | Autoresearch (Outer) | Self-Evolving (Inner) | This Repo |
|------------|---------------------|-----------------------|-----------|
| Structural code changes | Yes | No | Yes (outer loop) |
| Prompt optimization | Manual (coding agent) | Auto (metaprompt) | Auto + skill-aware |
| Skill learning | No | From successes only | From successes AND failures |
| Reflective memory | No | Episodic + semantic | Episodic + semantic |
| Failure analysis | No | Grader-based | Grader-based + defensive skill creation |
| State bridge | No | No | Yes (`evolution_state/`) |
| Plateau detection | No | Yes (5% threshold) | Yes + outer-loop handoff |
| Skill format | N/A | Basic SKILL.md | Anthropic skill-creator format |
| Experiment tracking | `results.tsv` | Metrics history | Both (`results.tsv` + metrics) |
| Git integration | Commit per experiment | No | Commit per outer-loop experiment |

---

## Persistence & File Layout

```
auto_research_self_improving_agents/
├── src/                              # Source code
│   ├── agent/                        # Agent construction
│   │   ├── deep_agent.py             # Factory: LLM + prompt + memory + skills → agent
│   │   ├── prompts.py                # All prompt templates
│   │   ├── prompt_store.py           # Versioned prompt persistence
│   │   └── subagents.py              # Research + synthesis sub-agents
│   ├── evolution/                    # Self-improvement engine
│   │   ├── orchestrator.py           # 10-node LangGraph evolution loop
│   │   ├── analyzer.py               # 3-axis trajectory grading
│   │   ├── skill_extractor.py        # Success → SKILL.md
│   │   ├── failure_skill_creator.py  # Failures → defensive SKILL.md
│   │   ├── prompt_optimizer.py       # Skill-aware prompt rewriting
│   │   ├── evolution_state_bridge.py # Inner → outer loop state files
│   │   ├── state.py                  # LangGraph state schemas
│   │   └── graders/                  # task_completion, efficiency, quality
│   ├── memory/                       # Dual memory system
│   │   ├── store.py                  # JSON-backed episodic + semantic
│   │   ├── reflection.py             # Post-run LLM introspection
│   │   └── compression.py            # Dedup + consolidation + budgeting
│   ├── skills/                       # SKILL.md management
│   │   └── manager.py                # CRUD + progressive disclosure
│   ├── tools/                        # Agent tools
│   │   └── search.py                 # Tavily web search
│   ├── tracing/                      # LangSmith integration
│   │   ├── fetcher.py                # Trace fetching (currently disabled)
│   │   └── trajectory.py             # Pydantic trajectory models
│   ├── config/                       # Configuration
│   │   └── settings.py               # Pydantic-settings env loading
│   └── cli/                          # CLI interface
│       └── commands.py               # run, evolve, prompts, skills, memory, state
│
├── evolution_state/                  # Bridge: inner → outer loop
│   ├── failures.md                   # Deduplicated failure patterns
│   ├── hypotheses.md                 # What's been tried + outcomes
│   ├── skills_summary.md             # Success + defensive skills
│   └── plateau_report.md             # Handoff signal
│
├── prompts/                          # Versioned prompts
│   └── v{NNNN}.json                  # {version, prompt, score, parent, feedback}
│
├── memory/                           # Persistent memory
│   ├── episodic/                     # One JSON per run
│   └── semantic/                     # Facts + patterns (UUID-keyed)
│
├── skills/                           # Learned skills
│   └── {skill-id}/SKILL.md           # YAML frontmatter + markdown body
│
├── tasks/research_tasks.json         # Evaluation dataset (7 tasks)
├── results.tsv                       # Outer-loop experiment log
├── program.md                        # Instructions for outer-loop coding agent
├── Makefile                          # Convenience commands
└── pyproject.toml                    # Dependencies + tool config
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Agent framework | `deepagents` | `create_deep_agent()` factory with skills middleware |
| LLM abstraction | `langchain` + `init_chat_model()` | Multi-provider support (20+) |
| Workflow engine | `langgraph` | StateGraph for orchestrator + analyzer |
| Web search | `langchain-tavily` | Advanced web search tool |
| Tracing | `langsmith` | Observability and experiment comparison |
| Configuration | `pydantic-settings` | Type-safe env var loading |
| Environment | `python-dotenv` | `.env` file support |

**Optional LLM providers**: `langchain-anthropic`, `langchain-google-genai`, `langchain-groq`, `langchain-ollama`
