# Auto-Research Self-Improving Agents

A self-improving research agent built on the **`evoagent`** library, combining three approaches:

1. **Karpathy's autoresearch** — outer-loop coding agent that makes structural code changes (new tools, model swaps, architecture) guided by eval results
2. **LangChain harness engineering** — middleware stack (self-verification, loop detection, context assembly, trace capture) that improves agent quality in real-time
3. **Letta's continual learning** — sleep-time compute, episodic/semantic memory, and skill extraction that improve the agent across sessions

## Architecture: Continuous Self-Improvement

The system self-improves through three modes of operation:

```
┌─────────────────────────────────────────────────────────────┐
│  OUTER LOOP — Karpathy autoresearch pattern                  │
│  Coding agent reads evolution_state/ and makes structural    │
│  changes when the inner loop plateaus                        │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  BACKGROUND DAEMON — Continuous self-evolution          │  │
│  │  Watches run_log.jsonl → optimizes prompt, extracts     │  │
│  │  skills, creates failure skills, compresses memory      │  │
│  │                                                        │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │  HARNESS LAYER — Real-time middleware             │  │  │
│  │  │  SelfVerification, ContextAssembly, LoopDetection │  │  │
│  │  │                                                  │  │  │
│  │  │  ┌────────────────────────────────────────────┐  │  │  │
│  │  │  │  AGENT — Think → Search → Synthesize       │  │  │  │
│  │  │  │  Every run: grade → score → reflect → log  │  │  │  │
│  │  │  └────────────────────────────────────────────┘  │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  SLEEP-TIME COMPUTE — Cross-run trace analysis → memory      │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.12+
- API keys: OpenAI (for the LLM) and Tavily (for web search, optional — DuckDuckGo is used as free fallback)
- Optional: LangSmith API key (for tracing)

## Quick Start

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the evoagent library (required — src/ depends on it)
pip install -e "./evoagent[all]"

# 3. Install the research agent application
pip install -e .

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys (at minimum: OPENAI_API_KEY)

# 5. Start the background evolution daemon (in a separate terminal)
python -m src evolve-daemon

# 6. Run research tasks — the agent self-improves from every interaction
python -m src run "What are the latest advances in quantum computing?"
python -m src run "Compare solid-state vs lithium-sulfur batteries"
python -m src run "What is the current state of fusion energy?"
# After 3+ runs, the daemon auto-evolves prompts, harness, and skills

# 7. (Optional) Bootstrap with synthetic tasks if starting fresh
python -m src evolve --tasks-file tasks/research_tasks.json --max-cycles 3

# 8. Run sleep-time review (cross-run trace analysis)
python -m src sleep-review
```

## How the Agent Self-Improves

### From every single user interaction

Every `python -m src run "..."` automatically:

1. **Grades** the output on 4 dimensions (task completion, efficiency, quality, claim verification)
2. **Scores** the current prompt version (incremental averaging across runs)
3. **Logs feedback** on failing dimensions to the prompt version
4. **Reflects** — LLM introspection stores episodic + semantic memories
5. **Appends** structured results to `evolution_state/run_log.jsonl`

### Background evolution daemon (primary)

The daemon (`python -m src evolve-daemon`) is the primary evolution mechanism. It runs alongside the serving agent and continuously improves from real user interactions:

```
┌─────────────┐     append      ┌──────────────┐
│  User runs   │ ──────────────→ │  run_log.jsonl │
│  (Terminal 1)│                 └──────┬───────┘
└─────────────┘                        │ poll (60s)
                                       ▼
                                ┌──────────────┐
                                │  Daemon       │  triggers when 3+ runs
                                │  (Terminal 2) │  have accumulated
                                └──────┬───────┘
                                       │ writes
                      ┌────────────┬───┴───┬─────────────┐
                      ▼            ▼       ▼             ▼
                 prompts/v*   harness/v*  skills/    memory/
                      │            │
                      ▼            ▼
              Next run picks up improved
              prompt AND evolved harness
```

Each daemon cycle optimizes 5 things:
1. **System prompt** — failure-aware, dimension-aware rewriting with pairwise validation
2. **Harness parameters** — middleware thresholds, retry counts, time budgets, reasoning effort (from raw execution traces)
3. **Success skills** — reusable patterns extracted from high-scoring runs
4. **Failure skills** — defensive "antibody" skills from low-scoring runs
5. **Memory compression** — deduplicates semantic memories

The daemon never re-runs tasks — it works entirely on grading results from real user interactions, making it more effective than synthetic benchmarks.

### Batch evolution (bootstrapping only)

The `evolve` command is useful for **initial setup** when you have no run history yet. Once the daemon is running and processing real user interactions, `evolve` is unnecessary.

```bash
# Only needed once, for bootstrapping
python -m src evolve --tasks-file tasks/research_tasks.json --max-cycles 3
```

## Running Tests

```bash
# All tests (259 total)
make test

# With coverage
make test-cov

# Specific test suites
.venv/bin/python -m pytest tests/evoagent/unit/ -v      # Library unit tests
.venv/bin/python -m pytest tests/evoagent/integration/ -v # Library integration
.venv/bin/python -m pytest tests/unit/ -v                # Application tests
.venv/bin/python -m pytest tests/evoagent/e2e/ -v        # E2E evolution loop
```

---

## EvoAgent Library

The generic self-improving patterns live in the **`evoagent`** library (in `evoagent/`). Team members can reuse these components to build their own self-improving agents — code review agents, security audit agents, content generators, etc.

### Install tiers

```bash
pip install -e ./evoagent                  # Core: memory, skills, config, types (pydantic only)
pip install -e "./evoagent[harness]"       # + middleware (adds langchain-core)
pip install -e "./evoagent[evolution]"     # + full Karpathy loop (adds langgraph, langsmith)
pip install -e "./evoagent[all]"           # Everything
```

### Cherry-pick components

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

### Full evolution loop

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

### Extending: custom graders

Implement the `Grader` protocol to add your own evaluation axes:

```python
from evoagent.core.protocols import Grader
from evoagent.core.types import GraderResult

class SecurityGrader(Grader):
    name = "security"

    def grade(self, task: str, output: str, **kwargs) -> GraderResult:
        # Your custom logic here
        return GraderResult(name="security", score=0.9, passed=True, reasoning="No issues found")

# Mix with built-in graders
graders = [EfficiencyGrader(), MultiJudgeGrader(llm, "quality"), SecurityGrader()]
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
├── harness/                      # Layer 2: middleware (needs langchain-core)
│   ├── middleware.py              # SelfVerification, ContextAssembly, LoopDetection, TraceCapture
│   └── builder.py                # default_middleware_stack()
├── graders/                      # Layer 2: evaluation (needs langchain-core)
│   ├── multi_judge.py            # MultiJudgeGrader (parallel LLM judges, median aggregation)
│   └── efficiency.py             # EfficiencyGrader (rule-based, no LLM)
├── evolution/                    # Layer 3: full loop (needs langgraph)
│   ├── analyzer.py               # Run graders, classify trajectories (with critical grader gate)
│   ├── prompt_optimizer.py       # Metaprompt-based prompt rewriting + autonomy validation
│   ├── sleep_review.py           # Cross-run trace analysis (sleep-time compute)
│   └── state.py                  # Evolution state persistence
└── tracing/                      # Trace capture + trajectory models
    └── trajectory.py             # TrajectoryRecord, ToolCall
```

---

## Research Agent (Reference Application)

The `src/` directory is a **reference application** built on `evoagent`. It demonstrates the full two-speed evolution pattern applied to research task automation.

### Continuous self-improvement (no explicit evolve needed)

Every `python -m src run "..."` grades, scores prompt + harness, reflects into memory, and logs results. The background daemon picks up accumulated results and evolves the system continuously:

- **Prompt optimization** — failure-aware, dimension-aware rewriting with pairwise validation
- **Harness evolution** — middleware parameters (retry counts, time budgets, loop thresholds, reasoning effort) tuned from raw execution traces
- **Skill extraction** — reusable patterns from successes, defensive "antibody" skills from failures
- **Memory compression** — semantic deduplication keeps context lean

No explicit `evolve` call needed — just run tasks and let the daemon learn.

### Bootstrapping with batch evolution (optional)

For initial setup when you have no run history, you can bootstrap with synthetic tasks:

```bash
python -m src evolve --tasks-file tasks/research_tasks.json --max-cycles 3
```

Once the daemon is running and processing real interactions, this is unnecessary.

### Viewing artifacts

```bash
python -m src skills        # List learned skills
python -m src prompts       # Show prompt version history (with scores from single runs)
python -m src memory        # Browse episodic and semantic memories
python -m src state         # Show evolution state and daemon activity
python -m src sleep-review  # Run cross-run trace analysis
```

| Directory          | Contents                                                                |
| ------------------ | ----------------------------------------------------------------------- |
| `skills/`          | SKILL.md files — success patterns + defensive "antibody" skills         |
| `prompts/`         | Versioned prompt JSON files with incremental scores and feedback        |
| `harness_config/`  | Versioned harness config JSON files with middleware parameter scores    |
| `memory/episodic/` | One JSON per agent run (task, strategy, score, what worked)             |
| `memory/semantic/` | Extracted facts and patterns (reusable across tasks)                    |
| `evolution_state/` | Daemon run log (`run_log.jsonl`) + outer loop bridge files             |

### Outer loop (structural changes)

The outer loop is for changes the daemon **cannot** make: adding new tools, swapping models, refactoring the grading pipeline, modifying agent architecture. It's driven by a coding agent (Claude Code, Cursor, Codex) that reads `evolution_state/` when the system plateaus.

See `program.md` for the full coding-agent instructions.

```bash
# 1. Create a branch for experimentation
git checkout -b autoresearch/<tag>

# 2. Run baseline inner loop
python -m src evolve --tasks-file tasks/research_tasks.json --max-cycles 5

# 3. Read what the inner loop learned
cat evolution_state/failures.md
cat evolution_state/plateau_report.md

# 4. Make structural changes, re-run, keep or revert
```

### Configuration

Key settings in `.env`:

```bash
MODEL=gpt-4o                          # Model for agent and graders
MODEL_PROVIDER=openai
TAVILY_API_KEY=tvly-...               # Optional (DuckDuckGo fallback is free)
MAX_EVOLUTION_CYCLES=5                # Inner loop cycles
BATCH_SIZE=3                          # Tasks sampled per evolution cycle
MEMORY_TOKEN_BUDGET=4000              # Memory context budget (chars / 4)
USE_SUBAGENTS=true                    # Enable research + synthesis sub-agents
```

### Search tool resilience

- **Primary**: Tavily (structured results, requires `TAVILY_API_KEY`)
- **Fallback**: DuckDuckGo (free, no API key needed)
- **Last resort**: Agent writes report from training knowledge

---

## Project Structure

```
evoagent/                             # Reusable library (see above)

src/                                  # Research agent application (uses evoagent)
├── agent/
│   ├── deep_agent.py                 # LangGraph agent with evoagent middleware
│   ├── prompts.py                    # Research-specific prompt + metaprompt templates
│   ├── prompt_store.py               # Versioned prompt persistence with incremental scoring
│   └── subagents.py                  # Research + synthesis sub-agents
├── evolution/
│   ├── orchestrator.py               # 11-node evolution loop with task sampling + holdout
│   ├── analyzer.py                   # 4-grader analysis with critical grader gate
│   ├── prompt_optimizer.py           # Dimension-aware prompt optimization with violation feedback
│   ├── daemon.py                     # Background evolution daemon (watches run_log)
│   ├── run_log.py                    # Append-only JSONL run log for daemon consumption
│   ├── evolution_state_bridge.py     # Bridge inner→outer loop
│   ├── state.py                      # LangGraph state schemas (imports evoagent types)
│   └── graders/                      # Research-specific graders
│       ├── claim_verification.py     # Claim extraction + internal consistency
│       └── fact_checker.py           # Web-based spot-check verification
├── memory/
│   └── reflection.py                 # Post-run LLM reflection
├── tools/
│   └── search.py                     # Tavily + DuckDuckGo fallback
├── tracing/
│   └── fetcher.py                    # LangSmith trace fetching
├── config/
│   └── settings.py                   # App-level settings
└── cli/
    └── commands.py                   # CLI: run, evolve, evolve-daemon, sleep-review, etc.

tests/
├── evoagent/                         # Library tests
│   ├── unit/                         # Types, parsing, config, memory, skills, efficiency
│   ├── integration/                  # Multi-judge grader, middleware, analyzer
│   └── e2e/                          # Full evolution loop with toy agent
└── unit/                             # Application tests
    ├── test_analyzer_parallel.py     # 4-grader analyzer + critical grader gate tests
    ├── test_claim_verification.py    # Claim verification tests
    ├── test_fact_checker.py          # Fact checker tests
    ├── test_multi_judge.py           # MultiJudgeGrader tests
    ├── test_pairwise.py              # Pairwise prompt comparison tests
    └── test_prompts.py               # Prompt template tests

evolution_state/                      # Runtime: bridge to outer-loop + run_log.jsonl
program.md                            # Instructions for the outer-loop coding agent
tasks/research_tasks.json             # Evaluation dataset
```

## Research Sources

This project synthesizes techniques from:

- [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — outer-loop evolution pattern
- [LangChain: Improving Deep Agents with Harness Engineering](https://blog.langchain.com/improving-deep-agents-with-harness-engineering/) — middleware stack design
- [Stanford Meta-Harness](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact) — environment bootstrapping
- [Letta: Continual Learning in Token Space](https://www.letta.com/blog/continual-learning) — memory and skill learning
- [Letta: Skill Learning for CLI Agents](https://www.letta.com/blog/skill-learning) — feedback-informed skill extraction
- [Harrison Chase: Continual Learning for AI Agents](https://x.com/hwchase17/article/2040467997022884194) — three-layer learning model
