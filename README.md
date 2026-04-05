# Auto-Research Self-Improving Agents

A self-improving research agent that combines three approaches:

1. **Karpathy's autoresearch** — outer-loop coding agent that makes structural code changes (new tools, model swaps, architecture) guided by eval results
2. **Self-evolving deep agents** — inner-loop self-evolution via prompt optimization, skill extraction, reflective memory, and plateau detection
3. **Anthropic's skill-creator** — structured SKILL.md format with assertive trigger descriptions, applied to both success-derived AND failure-derived skills

## Key Innovation: Two-Speed Evolution

```
┌─────────────────────────────────────────────────────┐
│              OUTER LOOP: Coding Agent                │
│         (Claude Code / Cursor / Codex)               │
│  Reads: program.md + evolution_state/ + results.tsv  │
│  Writes: agent code, tools, architecture             │
├─────────────────────────────────────────────────────┤
│              INNER LOOP: Evolution Engine             │
│  ┌──────────┐  ┌─────────┐  ┌───────────────────┐   │
│  │ Run Batch│→ │ Grade   │→ │ Reflect & Learn   │   │
│  │ (agent)  │  │ (3 axes)│  │ (memory + skills) │   │
│  └──────────┘  └─────────┘  └───────────────────┘   │
│        │                            │                │
│        │    ┌────────────────┐      │                │
│        └───→│ Failure Skills │←─────┘                │
│             │ (antibodies)   │                        │
│             └────────┬───────┘                        │
│                      ↓                                │
│             ┌────────────────┐                        │
│             │ Prompt Optimize│                        │
│             │ (skill-aware)  │                        │
│             └────────┬───────┘                        │
│                      ↓                                │
│             ┌────────────────┐                        │
│             │ Persist State  │→ evolution_state/      │
│             └────────────────┘                        │
├─────────────────────────────────────────────────────┤
│              EVAL: 3-Axis Graders                    │
│  task_completion (LLM) + efficiency (rule) + quality │
└─────────────────────────────────────────────────────┘
```

### Failure-Driven Skill Creation

The key differentiator: instead of only learning from successes, the system creates **defensive "antibody" skills** from failures. When the agent fails at a task, the failure skill creator:

1. Groups failures by pattern (which grader failed)
2. Analyzes the failure trajectories
3. Creates SKILL.md files with prevention strategies
4. These skills are automatically injected into future agent runs

This means the agent literally **immunizes itself** against recurring failure modes.

## Prerequisites

- Python 3.12+
- API keys: OpenAI (for the LLM) and Tavily (for web search, optional — DuckDuckGo is used as free fallback)
- Optional: LangSmith API key (for tracing)

## Quick Start

```bash
# 1. Create virtual environment and install dependencies
make install

# Or manually:
python3 -m venv .venv
source .venv/bin/activate   # On macOS/Linux
.venv\Scripts\activate      # On Windows
pip install -e .

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys (at minimum: OPENAI_API_KEY)

# 3. Run a single research task
make run TASK="What are the latest advances in quantum computing?"

# 4. Run the inner-loop evolution (see below for details)
make evolve CYCLES=3
```

## Running the Inner Loop

The inner loop is the automatic self-evolution engine. It runs the agent on a batch of research tasks, grades the output, reflects on what worked, extracts skills, and optimizes the prompt — all without human intervention.

### Basic usage

```bash
# Run 3 evolution cycles on the default 7-task research dataset
make evolve CYCLES=3
```

This is equivalent to:

```bash
.venv/bin/python -m src evolve --tasks-file tasks/research_tasks.json --max-cycles 3
```

### What happens during each cycle

The inner loop runs a **10-node LangGraph pipeline** per cycle:

```
1. run_batch          — Execute the agent on all tasks (5-min timeout per task)
2. fetch_traces       — Optionally enrich from LangSmith (currently uses local trajectories)
3. analyze            — Grade each trajectory: task_completion (LLM), efficiency (rules), quality (LLM)
4. reflect            — LLM generates introspection → stores episodic + semantic memories
5. compress_memories  — Deduplicate semantic memories, consolidate old episodic ones
6. extract_skills     — From successful runs (score >= 0.70) → creates SKILL.md files
7. create_failure_skills — From failure patterns (2+ failures) → creates defensive SKILL.md files
8. optimize_prompt    — Failure-aware + skill-aware prompt rewriting → new prompt version
9. persist_state      — Write evolution_state/ files for the outer loop
10. aggregate_metrics — Compute cycle summary, check for plateau (< 5% improvement × 2 cycles)
```

The loop continues until `max_cycles` is reached or a plateau is detected.

### Example output

```
============================================================
  CYCLE 0  |  Running 7 tasks  |  prompt v1
============================================================
[1/7] Running: What are the latest advances in quantum computing...
[1/7] Status: completed  (output: 4906 chars)
...
--- Analyze Trajectories ---
  [c2fa829b] What is quantum computing? -> SUCCESSFUL (avg=0.866)
  task_completion=0.80(PASS)  efficiency=0.97(PASS)  quality=0.83(PASS)
Analysis summary: 5 successful, 1 partial, 1 failed
--- Reflect & Store Memories ---
Memory totals: 7 episodic, 48 semantic
--- Extract Skills (from successes) ---
  NEW SKILL: structured-research-synthesis
Skills extracted this cycle: 1
--- Create Failure Skills (from failures) ---
  NEW DEFENSIVE SKILL: handle-task-completion-failures
Failure skills created this cycle: 1
--- Optimize Prompt (skill-aware) ---
Prompt upgraded: v1 -> v2 (1245 chars)
------------------------------------------------------------
  CYCLE 0 COMPLETE  |  avg_score=0.832  |  skills=1 (defensive=1)
  |  prompt=v2  |  memories=55
  -> Continuing to cycle 1
------------------------------------------------------------
```

### Viewing inner loop artifacts

After running the inner loop, inspect what it learned:

```bash
# List all learned skills (success-derived + defensive)
make skills

# Show prompt version history with scores
make prompts

# Browse episodic and semantic memories
make memory

# Show evolution state (what the outer loop reads)
make state
```

### Artifacts produced


| Directory          | Contents                                                                                      |
| ------------------ | --------------------------------------------------------------------------------------------- |
| `skills/`          | SKILL.md files — success patterns + defensive "antibody" skills                               |
| `prompts/`         | Versioned prompt JSON files (`v0001.json`, `v0002.json`, ...)                                 |
| `memory/episodic/` | One JSON per agent run (task, strategy, score, what worked)                                   |
| `memory/semantic/` | Extracted facts and patterns (reusable across tasks)                                          |
| `evolution_state/` | Human-readable state for the outer loop (`failures.md`, `hypotheses.md`, `skills_summary.md`) |


### Configuration

Key settings in `.env` that affect the inner loop:

```bash
# Model used by the agent and graders
MODEL=gpt-5.4-mini
MODEL_PROVIDER=openai

# Maximum inner-loop cycles (default: 5)
MAX_EVOLUTION_CYCLES=5

# Memory token budget — how much memory context to inject into prompts
# Lower = less noise, higher = more context (default: 1000)
MEMORY_TOKEN_BUDGET=1000

# Similarity threshold for deduplicating semantic memories (default: 0.7)
COMPRESSION_SIMILARITY_THRESHOLD=0.7

# Enable research + synthesis sub-agents (default: true)
USE_SUBAGENTS=true
```

### Search tool resilience

The agent uses a dual search strategy:

- **Primary**: Tavily (structured results, requires `TAVILY_API_KEY`)
- **Fallback**: DuckDuckGo (free, no API key needed, unlimited)

If Tavily hits its plan quota, the agent automatically falls back to DuckDuckGo. If both fail, the agent produces a report using its training knowledge.

### Troubleshooting the inner loop

**Agent asks for user input instead of producing a report:**
The system prompt has autonomy guardrails ("NEVER ask the user for input"). If this happens, check `prompts/` for the latest version — the prompt optimizer may have drifted. Reset to v0001.json or delete newer versions.

**Skills directory stays empty:**
Skills are only extracted from runs scoring >= 0.70. If the agent is producing poor output (e.g., search failures), scores will be low and no skills are created. Check that search is working (`make run TASK="test query"`).

**Inner loop hangs:**
Each task has a 5-minute timeout. If the loop seems stuck, check the logs — a task may be timing out. Reduce the number of tasks or simplify them.

**Prompt version never changes:**
The prompt optimizer only runs when there are failures. If all tasks succeed, the prompt stays the same (which is correct). The optimizer also has autonomy validation — if it generates a prompt that asks for user input, it falls back to the current prompt.

## Running the Outer Loop

The outer loop is a **greedy hill-climbing experiment loop** (inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)) driven by a coding agent (Claude Code, Cursor, Codex). It makes structural code changes that the inner loop cannot — adding tools, swapping models, or restructuring the agent architecture.

See `program.md` for the full coding-agent instructions.

### Setup

```bash
# 1. Create a branch for experimentation
git checkout -b autoresearch/<tag>

# 2. Read the key files
cat program.md                        # Your instructions
cat evolution_state/failures.md       # What the inner loop couldn't fix
cat evolution_state/plateau_report.md # Why the inner loop stopped

# 3. Run baseline inner loop
make evolve CYCLES=5

# 4. Initialize results.tsv with the baseline score
```

### The experiment loop

The outer loop runs for a bounded number of iterations (default: 5, set via `MAX_OUTER_LOOP_ITERATIONS` in `.env`).

Each iteration:

```
┌─────────────────────────────────────────────────────────┐
│  1. Run inner loop:  make evolve CYCLES=5               │
│  2. Read evolution_state/*.md                           │
│  3. Decide what structural change to make               │
│  4. Make ONE change, git commit                         │
│  5. Re-run inner loop to validate                       │
│  6. If score improved → KEEP commit, log to results.tsv │
│     If score same/worse → git reset --hard, log DISCARD │
│  7. Check stopping conditions                           │
└─────────────────────────────────────────────────────────┘
```

### Decision protocol

After reading `evolution_state/`, decide what to change:


| Failure pattern                                  | Action                                                  |
| ------------------------------------------------ | ------------------------------------------------------- |
| **Capability gaps** (agent can't do something)   | Add new tools to `src/tools/`                           |
| **Architectural issues** (wrong agent structure) | Restructure `src/agent/deep_agent.py` or `subagents.py` |
| **Model limitations** (quality ceiling)          | Swap model in `.env` (e.g. `gpt-4o` → `gpt-5-mini`)     |
| **Search quality** (poor sources)                | Improve search tool or add new search providers         |
| **All scores high** (nothing to fix)             | Expand `tasks/research_tasks.json` with harder examples |
| **Prompt-related** (wording issues)              | Skip — the inner loop handles prompt optimization       |


### What the coding agent CAN modify

- `src/agent/deep_agent.py` — agent construction, model selection, tool wiring
- `src/agent/prompts.py` — prompt templates (though inner loop also optimizes these)
- `src/agent/subagents.py` — sub-agent definitions
- `src/tools/` — add new tools, improve existing ones
- `src/config/settings.py` — configuration options
- `src/evolution/prompt_optimizer.py` — prompt validation guardrails
- `tasks/research_tasks.json` — expand the evaluation dataset
- `.env` — model selection, API keys, tuning parameters

### What the coding agent CANNOT modify

- `src/evolution/orchestrator.py` — the inner-loop pipeline is fixed
- `src/evolution/graders/` — grading functions are ground truth
- `src/evolution/analyzer.py` — analysis pipeline is fixed
- `evoagent/` — the library is shared infrastructure, not per-experiment

### Stopping conditions

The outer loop stops when any of these conditions are met:

- **Max iterations reached** (default: 5)
- **Score target**: avg_score >= 0.95 for 2 consecutive iterations
- **No progress**: 3 consecutive iterations where the best score doesn't improve
- **Manual interruption**: the human stops the coding agent

### Logging results

Each experiment is logged to `results.tsv` (tab-separated):

```
commit  overall_score  skills_count  failure_skills  prompt_version  status  description
9a2ddef  0.832         1             0               1               kept    Expand to 7 tasks
78c2228  0.848         1             0               1               kept    Reduce memory budget
004bd18  0.733         0             0               1               discard Search 10 results
```

### Example: running the outer loop with Claude Code

```bash
# Tell Claude Code to run the outer loop
# It reads program.md and follows the experiment loop autonomously

claude "Read program.md and run the outer loop experiment.
       Start by running the inner loop baseline, then make
       structural improvements based on the failure analysis."
```

### Known risks

**Prompt drift**: The inner loop's prompt optimizer can generate prompts that ask for user input. Guardrails in `prompt_optimizer.py` prevent this (autonomy validation + retry + fallback). See the "Known risks" section in `program.md`.

**Token cost**: Each outer-loop iteration runs the full inner loop (up to 5 cycles × 7 tasks = 35 agent runs + grading + reflection). With defaults (5 outer × 5 inner × 7 tasks), worst case is ~175 agent runs. Set `MAX_OUTER_LOOP_ITERATIONS` and `MAX_EVOLUTION_CYCLES` lower to reduce cost.

## EvoAgent Library

The generic self-improving patterns have been extracted into the **`evoagent`** library (in `evoagent/`). Team members can reuse these components to build their own self-improving agents — code review agents, security audit agents, content generators, etc.

### Install tiers

```bash
pip install evoagent                  # Core: memory, skills, config, types (pydantic only)
pip install evoagent[harness]         # + middleware (adds langchain)
pip install evoagent[evolution]       # + full Karpathy loop (adds langgraph, langsmith)
pip install evoagent[all]             # Everything
```

### Quick start: cherry-pick components

```python
# Just memory
from evoagent.memory import FileMemoryStore, compress_context
store = FileMemoryStore("./my-agent/memory")
store.store("episodic", "run-001", {"task": "...", "score": 0.85})

# Just middleware (add to any LangChain agent)
from evoagent.harness import SelfVerificationMiddleware, LoopDetectionMiddleware
middleware = [
    SelfVerificationMiddleware(required_sections=["summary", "recommendation"]),
    LoopDetectionMiddleware(max_similar=3),
]

# Just grading
from evoagent.graders import EfficiencyGrader, MultiJudgeGrader
grader = EfficiencyGrader()
result = grader.grade(task="...", output="...", metrics=my_metrics)
```

### Quick start: full evolution loop

```python
from evoagent import EvoAgentConfig
from evoagent.core.protocols import AgentFactory
from evoagent.core.types import TaskResult, TrajectoryMetrics
from evoagent.graders import EfficiencyGrader, MultiJudgeGrader
from evoagent.harness import default_middleware_stack
from evoagent.memory import FileMemoryStore
from evoagent.skills import SkillManager
from evoagent.evolution import analyze_trajectory, classify_trajectory, run_sleep_review

# 1. Implement AgentFactory for YOUR agent
class MyAgent(AgentFactory):
    def create(self, system_prompt, middleware, **kwargs):
        # Build your LangChain/LangGraph agent here
        ...

    def run(self, agent, task, timeout=300):
        # Run the agent and return a TaskResult
        ...

# 2. Configure
config = EvoAgentConfig(base_dir="./data", max_cycles=5)
memory = FileMemoryStore(config.memory_path)
skills = SkillManager(config.skills_path)

# 3. Set up graders (mix built-in + custom)
graders = [
    MultiJudgeGrader(llm=my_llm, name="task_completion"),
    MultiJudgeGrader(llm=my_llm, name="quality"),
    EfficiencyGrader(),
]

# 4. Run and grade
factory = MyAgent()
agent = factory.create(system_prompt="...", middleware=default_middleware_stack())
result = factory.run(agent, task="Research quantum computing")
grades = analyze_trajectory(graders, result.task, result.output,
                            metrics=TrajectoryMetrics(total_tokens=5000, total_steps=3))
classification, score = classify_trajectory(grades)

# 5. Sleep-time review (between sessions)
run_sleep_review(llm=my_llm, memory=memory, traces_dir=config.traces_path)
```

### Running the research agent example

This repo's `src/` is itself a reference application built on `evoagent`:

```bash
# 1. Install everything
pip install -e .

# 2. Configure
cp .env.example .env  # add your API keys

# 3. Run a single research task
make run TASK="What are the latest advances in quantum computing?"

# 4. Run the evolution loop
make evolve CYCLES=3

# 5. Run sleep-time review (cross-run trace analysis)
.venv/bin/python -m src sleep-review

# 6. Inspect what it learned
make skills     # List learned skills
make prompts    # Show prompt version history
make memory     # Browse memories
make state      # Show evolution state
```

### Running tests

```bash
# All tests (188 total)
PYTHONPATH=. python -m pytest tests/evoagent/ tests/unit/ -v

# Evoagent library only (52 tests, no API keys needed)
PYTHONPATH=. python -m pytest tests/evoagent/ -v

# Research agent tests (136 tests, mocked LLM)
PYTHONPATH=. python -m pytest tests/unit/ -v

# E2E grader test with real LLM (needs API keys in .env)
PYTHONPATH=. python test_graders.py
```

### Library architecture

```
evoagent/                         # Reusable library
├── core/                         # Layer 0: types, protocols, config (pydantic only)
│   ├── types.py                  # GraderResult, TaskResult, TrajectoryMetrics
│   ├── protocols.py              # ABCs: Grader, AgentFactory, MemoryBackend, SkillStore, PromptStore
│   ├── config.py                 # EvoAgentConfig
│   └── parsing.py                # LLM JSON response parsing
├── memory/                       # Layer 1: episodic/semantic memory
│   ├── store.py                  # FileMemoryStore
│   └── compression.py            # Token-budgeted context assembly + dedup
├── skills/                       # Layer 1: SKILL.md management
│   ├── manager.py                # SkillManager
│   └── extractor.py              # Success + failure skill extraction
├── harness/                      # Layer 2: middleware (needs langchain)
│   ├── middleware.py              # SelfVerification, ContextAssembly, LoopDetection, TraceCapture
│   └── builder.py                # default_middleware_stack()
├── graders/                      # Layer 2: evaluation (needs langchain)
│   ├── multi_judge.py            # MultiJudgeGrader (parallel LLM judges)
│   └── efficiency.py             # EfficiencyGrader (rule-based)
├── evolution/                    # Layer 3: full loop (needs langgraph)
│   ├── analyzer.py               # Run graders, classify trajectories
│   ├── prompt_optimizer.py       # Metaprompt-based prompt rewriting
│   ├── sleep_review.py           # Cross-run trace analysis
│   └── state.py                  # Evolution state persistence
└── tracing/                      # Trace capture + trajectory models
```

## Project Structure

```
src/                              # Research agent application (uses evoagent)
├── agent/                        # Agent factory, prompts, sub-agents
│   ├── deep_agent.py             # LangGraph agent with evoagent middleware
│   ├── prompts.py                # Research-specific prompt templates
│   ├── prompt_store.py           # Versioned prompt persistence
│   └── subagents.py              # Research + synthesis sub-agents
├── evolution/                    # Research-specific evolution pipeline
│   ├── orchestrator.py           # Two-speed evolution loop (10 LangGraph nodes)
│   ├── analyzer.py               # 4-grader analysis using evoagent graders
│   ├── prompt_optimizer.py       # Skill-aware prompt optimization
│   ├── evolution_state_bridge.py # Bridge inner→outer loop
│   ├── state.py                  # LangGraph state schemas (imports evoagent types)
│   └── graders/                  # Research-specific graders
│       ├── quality.py            # Output quality (uses evoagent parsing)
│       ├── task_completion.py    # Task completion (uses evoagent parsing)
│       ├── claim_verification.py # Claim extraction + internal consistency
│       └── fact_checker.py       # Web-based spot-check verification
├── memory/                       # Delegates to evoagent.memory
│   └── reflection.py             # Post-run LLM reflection (research-specific prompts)
├── tools/                        # Research-specific tools
│   └── search.py                 # Tavily (primary) + DuckDuckGo (fallback)
├── tracing/                      # Delegates to evoagent.tracing
│   └── fetcher.py                # LangSmith trace fetching
├── config/                       # App-level settings (pydantic-settings)
└── cli/                          # CLI commands (run, evolve, sleep-review, etc.)

evoagent/                         # Reusable library (see above)

evolution_state/                  # Bridge to outer-loop coding agent
├── failures.md                   # Deduplicated failure patterns
├── hypotheses.md                 # What's been tried + outcomes
├── skills_summary.md             # All skills (success + defensive)
└── plateau_report.md             # Handoff signal to outer loop

program.md                        # Instructions for the outer-loop coding agent
tasks/research_tasks.json         # Evaluation dataset (7 research tasks)
results.tsv                       # Experiment log (autoresearch style)
```

## How It Differs from Each Parent


| Feature             | autoresearch          | self_evolving_deep_agents | This repo                      |
| ------------------- | --------------------- | ------------------------- | ------------------------------ |
| Code changes        | Yes (outer loop)      | No                        | Yes (outer loop)               |
| Prompt optimization | Manual (coding agent) | Auto (metaprompt)         | Auto + skill-aware             |
| Skill learning      | No                    | From successes only       | From successes AND failures    |
| Memory              | No                    | Episodic + semantic       | Episodic + semantic            |
| Failure analysis    | No                    | Grader-based              | Grader-based + skill creation  |
| State bridge        | No                    | No                        | Yes (evolution_state/)         |
| Plateau detection   | No                    | Yes (5% threshold)        | Yes + outer-loop handoff       |
| Skill format        | N/A                   | Basic SKILL.md            | Anthropic skill-creator format |
| Search resilience   | N/A                   | Tavily only               | Tavily + DuckDuckGo fallback   |
| Outer loop bounds   | Runs forever          | N/A                       | Configurable max iterations    |


