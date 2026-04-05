# Auto-Research Self-Improving Agents

A self-improving research agent built on the **`evoagent`** library, combining three approaches:

1. **Karpathy's autoresearch** — outer-loop coding agent that makes structural code changes (new tools, model swaps, architecture) guided by eval results
2. **LangChain harness engineering** — middleware stack (self-verification, loop detection, context assembly, trace capture) that improves agent quality in real-time
3. **Letta's continual learning** — sleep-time compute, episodic/semantic memory, and skill extraction that improve the agent across sessions

## Architecture: 4-Layer Self-Improvement

```
┌───────────────────────────────────────────────────────────┐
│  OUTER LOOP — Karpathy autoresearch pattern               │
│  (prompt_optimizer.py, analyzer.py, orchestrator)         │
│                                                           │
│  Propose new prompt → Run eval tasks → Grade → Keep?      │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  HARNESS LAYER — LangChain/Stanford Meta-Harness    │  │
│  │  (evoagent.harness.middleware)                       │  │
│  │                                                     │  │
│  │  SelfVerification, ContextAssembly, LoopDetection   │  │
│  │  TraceCaptureMiddleware → feeds back to outer loop   │  │
│  │                                                     │  │
│  │  ┌───────────────────────────────────────────────┐  │  │
│  │  │  INNER LOOP — The agent itself                │  │  │
│  │  │  (deep_agent.py, LLM calls, tool use)         │  │  │
│  │  │                                               │  │  │
│  │  │  Think → Search → Synthesize → Write report   │  │  │
│  │  └───────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                           │
│  SLEEP-TIME COMPUTE — Letta continual learning            │
│  (evoagent.evolution.sleep_review)                        │
│  Cross-run analysis → meta-instructions → memory          │
└───────────────────────────────────────────────────────────┘
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

# 5. Run a single research task
python -m src run "What are the latest advances in quantum computing?"

# 6. Run the evolution loop (3 cycles)
python -m src evolve --tasks-file tasks/research_tasks.json --max-cycles 3

# 7. Run sleep-time review (cross-run trace analysis)
python -m src sleep-review
```

## Running Tests

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
│   ├── analyzer.py               # Run graders, classify trajectories
│   ├── prompt_optimizer.py       # Metaprompt-based prompt rewriting + autonomy validation
│   ├── sleep_review.py           # Cross-run trace analysis (sleep-time compute)
│   └── state.py                  # Evolution state persistence
└── tracing/                      # Trace capture + trajectory models
    └── trajectory.py             # TrajectoryRecord, ToolCall
```

---

## Research Agent (Reference Application)

The `src/` directory is a **reference application** built on `evoagent`. It demonstrates the full two-speed evolution pattern applied to research task automation.

### Failure-driven skill creation

The key differentiator: instead of only learning from successes, the system creates **defensive "antibody" skills** from failures. When the agent fails at a task, the skill extractor:

1. Groups failures by pattern (which grader failed)
2. Analyzes the failure trajectories
3. Creates SKILL.md files with prevention strategies
4. These skills are automatically injected into future agent runs

### Running the inner loop

The inner loop is the automatic self-evolution engine. It runs the agent on research tasks, grades the output, reflects, extracts skills, and optimizes the prompt — all without human intervention.

```bash
# Run 3 evolution cycles on the default research dataset
python -m src evolve --tasks-file tasks/research_tasks.json --max-cycles 3
```

Each cycle runs a **10-node LangGraph pipeline**:

```
1. run_batch          — Execute the agent on all tasks (5-min timeout per task)
2. fetch_traces       — Optionally enrich from LangSmith
3. analyze            — Grade each trajectory: task_completion, efficiency, quality, claims
4. reflect            — LLM generates introspection → stores episodic + semantic memories
5. compress_memories  — Deduplicate semantic memories
6. extract_skills     — From successful runs (score >= 0.70) → creates SKILL.md files
7. create_failure_skills — From failure patterns (2+ failures) → defensive SKILL.md files
8. optimize_prompt    — Failure-aware + skill-aware prompt rewriting → new prompt version
9. persist_state      — Write evolution_state/ files for the outer loop
10. aggregate_metrics — Compute cycle summary, check for plateau (< 5% improvement x 2 cycles)
```

### Viewing artifacts

```bash
python -m src skills      # List learned skills
python -m src prompts     # Show prompt version history
python -m src memory      # Browse episodic and semantic memories
python -m src state       # Show evolution state
python -m src sleep-review  # Run cross-run trace analysis
```

| Directory          | Contents                                                                |
| ------------------ | ----------------------------------------------------------------------- |
| `skills/`          | SKILL.md files — success patterns + defensive "antibody" skills         |
| `prompts/`         | Versioned prompt JSON files (`v0001.json`, `v0002.json`, ...)           |
| `memory/episodic/` | One JSON per agent run (task, strategy, score, what worked)             |
| `memory/semantic/` | Extracted facts and patterns (reusable across tasks)                    |
| `evolution_state/` | Human-readable state for the outer loop (`failures.md`, `hypotheses.md`)|

### Running the outer loop

The outer loop is a **greedy hill-climbing experiment loop** (inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)) driven by a coding agent (Claude Code, Cursor, Codex). It makes structural code changes that the inner loop cannot.

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

Each iteration:

```
1. Run inner loop → read evolution_state/*.md
2. Decide what structural change to make
3. Make ONE change, git commit
4. Re-run inner loop to validate
5. If score improved → KEEP commit
   If score same/worse → git reset --hard
```

### Configuration

Key settings in `.env`:

```bash
MODEL=gpt-4o                          # Model for agent and graders
MODEL_PROVIDER=openai
TAVILY_API_KEY=tvly-...               # Optional (DuckDuckGo fallback is free)
MAX_EVOLUTION_CYCLES=5                # Inner loop cycles
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
│   ├── prompts.py                    # Research-specific prompt templates
│   ├── prompt_store.py               # Versioned prompt persistence
│   └── subagents.py                  # Research + synthesis sub-agents
├── evolution/
│   ├── orchestrator.py               # Two-speed evolution loop (10 LangGraph nodes)
│   ├── analyzer.py                   # 4-grader analysis using evoagent graders
│   ├── prompt_optimizer.py           # Skill-aware prompt optimization
│   ├── evolution_state_bridge.py     # Bridge inner→outer loop
│   ├── state.py                      # LangGraph state schemas (imports evoagent types)
│   └── graders/                      # Research-specific graders
│       ├── quality.py                # Output quality
│       ├── task_completion.py        # Task completion
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
    └── commands.py                   # CLI: run, evolve, sleep-review, etc.

tests/
├── evoagent/                         # Library tests (52 tests)
│   ├── unit/                         # Types, parsing, config, memory, skills, efficiency
│   ├── integration/                  # Multi-judge grader, middleware, analyzer
│   └── e2e/                          # Full evolution loop with toy agent
└── unit/                             # Application tests (136 tests)
    ├── test_multi_judge.py           # MultiJudgeGrader tests
    ├── test_claim_verification.py    # Claim verification tests
    ├── test_fact_checker.py          # Fact checker tests
    ├── test_analyzer_parallel.py     # 4-grader analyzer tests
    ├── test_pairwise.py              # Pairwise prompt comparison tests
    └── test_prompts.py               # Prompt template tests

evolution_state/                      # Bridge to outer-loop coding agent
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
