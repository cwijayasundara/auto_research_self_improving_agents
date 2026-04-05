# EvoAgent Library Design Spec

**Date:** 2026-04-05
**Status:** Approved
**Author:** cwijayasundara + Claude

## Problem

The self-improving research agent codebase contains ~85% domain-agnostic patterns (middleware, memory, grading, evolution loop, skill extraction, sleep-time compute). These patterns implement well-established techniques from Karpathy's autoresearch (outer loop), LangChain harness engineering (middleware), Stanford Meta-Harness (environment bootstrapping), and Letta's continual learning (token-space learning). Team members building other agent types (code review, security audit, content generation) need to reuse these patterns without forking the research agent.

## Solution

A single Python package `evoagent` with optional dependency extras. Every component is usable independently (cherry-pick) or composed into the full Karpathy-style evolution loop. Pluggability is achieved through protocol classes (ABCs) with shipped default implementations.

## Approach

Single package with optional extras (Approach C):
- `pip install evoagent` — core: memory, skills, config, types (2 deps: pydantic)
- `pip install evoagent[harness]` — + middleware, graders (adds langchain)
- `pip install evoagent[evolution]` — + full loop (adds langgraph, langsmith)
- `pip install evoagent[all]` — everything

---

## Package Structure

```
evoagent/
├── core/               # Layer 0 — zero heavy deps (just pydantic)
│   ├── __init__.py
│   ├── types.py        # GraderResult, TaskResult, TrajectoryMetrics, EvolutionCycleReport
│   ├── protocols.py    # ABCs: Grader, AgentFactory, MemoryBackend, SkillStore, PromptStore, MiddlewareHook
│   ├── config.py       # EvoAgentConfig (Pydantic Settings base)
│   └── parsing.py      # LLM JSON response parsing utilities
│
├── memory/             # Layer 1 — episodic/semantic memory system
│   ├── __init__.py
│   ├── store.py        # MemoryStore (default: file-backed JSON, implements MemoryBackend)
│   ├── reflection.py   # Post-run reflection generator (configurable prompt)
│   ├── compression.py  # Token-budgeted context assembly + dedup + meta-instruction priority
│   └── backends/       # Future: sqlite, redis
│       └── __init__.py
│
├── skills/             # Layer 1 — SKILL.md management
│   ├── __init__.py
│   ├── manager.py      # Discover, create, validate, progressive disclosure (implements SkillStore)
│   └── extractor.py    # Extract skills from success + failure trajectories (unified)
│
├── harness/            # Layer 2 — middleware stack (needs langchain)
│   ├── __init__.py
│   ├── middleware.py    # SelfVerification, ContextAssembly, LoopDetection, TraceCapture
│   ├── prompts.py      # Middleware prompt templates (revision, loop warning, etc.)
│   └── builder.py      # default_middleware_stack() factory
│
├── graders/            # Layer 2 — evaluation system (needs langchain)
│   ├── __init__.py
│   ├── multi_judge.py  # Parallel perspective judging (configurable axis + prompts)
│   ├── efficiency.py   # Rule-based efficiency scoring
│   ├── claim_verify.py # Claim extraction + internal consistency checking
│   └── fact_check.py   # Web-based spot-check verification (optional search tool)
│
├── tracing/            # Layer 2 — trace capture & analysis
│   ├── __init__.py
│   ├── capture.py      # Trace recording to disk
│   ├── fetcher.py      # LangSmith trace fetching
│   └── trajectory.py   # Trajectory data models
│
├── evolution/          # Layer 3 — the full loop (needs langgraph)
│   ├── __init__.py
│   ├── loop.py         # EvolutionLoop: propose -> run -> grade -> accept/reject
│   ├── analyzer.py     # Parallel trajectory analysis (accepts list[Grader])
│   ├── prompt_optimizer.py  # Metaprompt-based prompt rewriting (configurable template)
│   ├── state.py        # Evolution state persistence & checkpointing
│   └── sleep_review.py # Cross-run trace analysis (sleep-time compute)
│
└── __init__.py         # Top-level convenience imports
```

### Dependency Layers

```
Layer 0 (core)                    -> pydantic only
Layer 1 (memory, skills)          -> core
Layer 2 (harness, graders, tracing) -> core + langchain
Layer 3 (evolution)               -> core + langchain + langgraph
```

---

## Core Protocols

Defined in `evoagent/core/protocols.py`. Every higher layer programs against these, not concrete implementations.

### Grader

```python
class Grader(ABC):
    """Evaluate one axis of agent output quality."""
    name: str

    @abstractmethod
    def grade(self, task: str, output: str, **kwargs) -> GraderResult: ...
```

### AgentFactory

```python
class AgentFactory(ABC):
    """Build and invoke the agent under evaluation."""

    @abstractmethod
    def create(self, system_prompt: str, middleware: list, **kwargs) -> Any: ...

    @abstractmethod
    def run(self, agent: Any, task: str, timeout: int) -> TaskResult: ...
```

### MemoryBackend

```python
class MemoryBackend(ABC):
    """Storage backend for episodic/semantic memories."""

    @abstractmethod
    def store(self, namespace: str, key: str, data: dict) -> None: ...

    @abstractmethod
    def retrieve(self, namespace: str, key: str) -> dict | None: ...

    @abstractmethod
    def list_all(self, namespace: str) -> dict[str, dict]: ...

    @abstractmethod
    def search(self, namespace: str, query: str, limit: int) -> list[dict]: ...
```

### SkillStore

```python
class SkillStore(ABC):
    """CRUD for skill files."""

    @abstractmethod
    def discover(self) -> dict[str, dict]: ...      # name -> {description, path}

    @abstractmethod
    def load(self, name: str) -> str: ...            # full skill content

    @abstractmethod
    def create(self, name: str, content: str) -> Path: ...
```

### PromptStore

```python
class PromptStore(ABC):
    """Versioned prompt persistence."""

    @abstractmethod
    def get_current(self) -> tuple[int, str]: ...     # (version, content)

    @abstractmethod
    def save(self, content: str, score: float, parent: int) -> int: ...  # returns new version
```

### MiddlewareHook

```python
class MiddlewareHook(ABC):
    """Single middleware in the harness stack."""

    def before_model(self, state, runtime) -> dict | None: ...
    def after_model(self, state, runtime) -> dict | None: ...
    def wrap_tool_call(self, request, handler) -> Any: ...
```

---

## Core Types

Defined in `evoagent/core/types.py`. Plain dataclasses, no framework dependencies.

```python
@dataclass
class GraderResult:
    name: str
    score: float            # 0.0 - 1.0
    passed: bool
    reasoning: str

@dataclass
class TaskResult:
    task: str
    output: str
    tool_calls: list[dict]
    duration_seconds: float
    status: Literal["success", "error", "timeout"]

@dataclass
class TrajectoryMetrics:
    total_tokens: int
    total_steps: int
    latency_seconds: float
    tool_call_count: int

@dataclass
class EvolutionCycleReport:
    cycle: int
    avg_score: float
    classification_counts: dict[str, int]
    prompt_version: int
    skills_created: int
    improvements: list[str]
```

---

## User-Facing API

### Tier 1: Cherry-Pick Components

```python
# Memory only
from evoagent.memory import MemoryStore, compress_context
store = MemoryStore("/tmp/my-agent/memory")
store.store("episodic", "run-001", {"task": "...", "score": 0.85})
context = compress_context(store, task="...", token_budget=4000)

# Skills only
from evoagent.skills import SkillManager
skills = SkillManager("/tmp/my-agent/skills")
skills.create("error-recovery", "---\nname: error-recovery\n...\n---\n# ...")

# Middleware only
from evoagent.harness import SelfVerificationMiddleware, LoopDetectionMiddleware
middleware = [
    SelfVerificationMiddleware(required_sections=["summary", "recommendation"], max_retries=2),
    LoopDetectionMiddleware(max_similar=3, max_total=12),
]
```

### Tier 2: Grading Pipeline Without the Loop

```python
from evoagent.graders import MultiJudgeGrader, EfficiencyGrader
from evoagent.evolution import analyze_trajectory, classify_trajectory

graders = [
    MultiJudgeGrader(llm=my_llm, axis="task_completion"),
    MultiJudgeGrader(llm=my_llm, axis="quality"),
    EfficiencyGrader(),
]

# Custom grader
class SecurityGrader(Grader):
    name = "security"
    def grade(self, task, output, **kwargs) -> GraderResult:
        return GraderResult(name="security", score=0.9, passed=True, reasoning="...")

graders.append(SecurityGrader())
results = analyze_trajectory(graders, task="...", output="...", metrics=my_metrics)
classification, avg_score = classify_trajectory(results)
```

### Tier 3: Full Evolution Loop

```python
from evoagent import EvolutionLoop, EvoAgentConfig
from evoagent.harness import default_middleware_stack
from evoagent.graders import MultiJudgeGrader, EfficiencyGrader
from evoagent.memory import MemoryStore
from evoagent.skills import SkillManager

class MyCodeReviewAgent(AgentFactory):
    def create(self, system_prompt, middleware, **kwargs):
        return create_deep_agent(model=my_llm, tools=my_tools,
                                 system_prompt=system_prompt, middleware=middleware)

    def run(self, agent, task, timeout=300):
        result = agent.invoke({"messages": [HumanMessage(task)]})
        return TaskResult(task=task, output=extract(result), ...)

loop = EvolutionLoop(
    agent_factory=MyCodeReviewAgent(),
    tasks=["Review this PR for security issues...", ...],
    graders=[MultiJudgeGrader(llm, "task_completion"), EfficiencyGrader()],
    memory=MemoryStore("./memory"),
    skills=SkillManager("./skills"),
    middleware=default_middleware_stack(),
    config=EvoAgentConfig(max_cycles=10, min_improvement_threshold=0.05),
)
report = loop.run()
```

---

## Extraction Mapping

### Current File -> Library Location

| Current File | Library Location | Changes |
|---|---|---|
| `src/config/settings.py` | `evoagent/core/config.py` | Strip research defaults; base class |
| `src/evolution/state.py` (types) | `evoagent/core/types.py` | Extract dataclasses |
| (new) | `evoagent/core/protocols.py` | ABCs from this spec |
| (new) | `evoagent/core/parsing.py` | Deduplicate JSON parsing (4 files today) |
| `src/memory/store.py` | `evoagent/memory/store.py` | Implement MemoryBackend protocol |
| `src/memory/reflection.py` | `evoagent/memory/reflection.py` | Make reflection prompt a parameter |
| `src/memory/compression.py` | `evoagent/memory/compression.py` | Near-verbatim |
| `src/skills/manager.py` | `evoagent/skills/manager.py` | Implement SkillStore protocol |
| `src/evolution/skill_extractor.py` | `evoagent/skills/extractor.py` | Merge success + failure extraction |
| `src/evolution/failure_skill_creator.py` | (merged into extractor.py) | Unify; configurable prompts |
| `src/agent/middleware.py` | `evoagent/harness/middleware.py` | Constructor params for patterns/sections |
| (new) | `evoagent/harness/builder.py` | `default_middleware_stack()` factory |
| `src/evolution/graders/multi_judge.py` | `evoagent/graders/multi_judge.py` | Implement Grader protocol; configurable prompts |
| `src/evolution/graders/efficiency.py` | `evoagent/graders/efficiency.py` | Near-verbatim |
| `src/evolution/graders/claim_verification.py` | `evoagent/graders/claim_verify.py` | Near-verbatim |
| `src/evolution/graders/fact_checker.py` | `evoagent/graders/fact_check.py` | Search tool as parameter |
| `src/evolution/orchestrator.py` | `evoagent/evolution/loop.py` | Replace hardcoded agent/graders with protocols |
| `src/evolution/analyzer.py` | `evoagent/evolution/analyzer.py` | Accept `list[Grader]` |
| `src/evolution/prompt_optimizer.py` | `evoagent/evolution/prompt_optimizer.py` | Configurable metaprompt template |
| `src/evolution/state.py` (persistence) | `evoagent/evolution/state.py` | Keep persistence logic |
| `src/agent/sleep_review.py` | `evoagent/evolution/sleep_review.py` | Near-verbatim |
| `src/tracing/trajectory.py` | `evoagent/tracing/trajectory.py` | Near-verbatim |
| `src/tracing/fetcher.py` | `evoagent/tracing/fetcher.py` | Near-verbatim |

### Key Refactoring Themes

1. **Hardcoded prompts -> constructor parameters** with shipped defaults
2. **Hardcoded lists -> protocol-typed lists** (`graders: list[Grader]`)
3. **Duplicate JSON parsing -> `core/parsing.py`** (exists in 4+ files)
4. **Merge success + failure skill extraction** into unified `extractor.py`

### What Stays Behind (Reference Application)

```
examples/research_agent/
├── agent.py            # AgentFactory implementation (from deep_agent.py)
├── prompts.py          # Research-specific prompts (DEFAULT_SYSTEM_PROMPT, etc.)
├── tools.py            # ResilientSearch (from tools/search.py)
├── subagents.py        # Research + synthesis sub-agents
├── graders.py          # Research-specific grader config
├── tasks.json          # Sample evaluation tasks
├── program.md          # Outer-loop steering document
└── README.md
```

---

## Dependencies

```toml
[project]
name = "evoagent"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
]

[project.optional-dependencies]
harness = ["langchain>=0.3", "langchain-community>=0.4"]
evolution = ["evoagent[harness]", "langgraph>=1.0", "langsmith>=0.7"]
all = ["evoagent[evolution]"]
dev = ["pytest>=9.0", "pytest-cov>=7.0", "ruff>=0.15", "mypy>=1.19"]
```

Import guards at each layer boundary:

```python
# evoagent/harness/__init__.py
try:
    from langchain_core.messages import BaseMessage
except ImportError:
    raise ImportError(
        "evoagent.harness requires langchain. Install with: pip install evoagent[harness]"
    ) from None
```

---

## Testing Strategy

### Unit Tests (Layer 0-1, no LLM)

- Memory store/retrieve/search with `tmp_path`
- Compression budget enforcement
- Skill discovery and SKILL.md parsing
- Type validation
- JSON parsing edge cases

### Integration Tests (Layer 2, mocked LLM)

- Multi-judge aggregation (median, fallback)
- Self-verification retry injection
- Loop detection warning trigger
- Trace capture file writing

### E2E Tests (Layer 3, toy agent)

- Full evolution loop with `ToyAgent(AgentFactory)` and `EfficiencyGrader` (deterministic)
- Verify cycle count, memory persistence, skill creation, prompt versioning
- Sleep-time review with synthetic traces

### CI Matrix

- Core + unit tests: every PR (fast, no API keys)
- Integration tests: every PR (mocked LLM, medium speed)
- E2E tests: nightly or release branches (optional real LLM)

---

## Documentation & Examples

```
docs/
├── getting-started.md              # Install, hello-world per tier
├── concepts.md                     # 4-layer model (inner loop, harness, sleep-time, outer loop)
├── guides/
│   ├── custom-grader.md            # Implement Grader in 20 lines
│   ├── custom-agent-factory.md     # Plug in your own agent
│   ├── memory-backends.md          # File, SQLite, or roll your own
│   └── full-loop.md               # Run the Karpathy loop end-to-end
├── api/                            # Auto-generated from docstrings
└── architecture.md                 # Protocol dependency diagram

examples/
├── research_agent/                 # Current app migrated to evoagent
├── code_review_agent/              # Custom AgentFactory + SecurityGrader
├── cherry_pick_memory/             # Just MemoryStore + compression
└── cherry_pick_middleware/         # Just middleware on existing agent
```

Each example is runnable and under 100 lines of user code.

---

## Design Principles

1. **Protocols over config** — pluggability via ABCs, not YAML/JSON schemas
2. **Defaults for everything** — team members implement only what they customize
3. **Layered dependencies** — core is 2 deps; pay for what you use
4. **Current app is just an example** — validates the extraction lost nothing
5. **Import guards** — clear error messages for missing optional deps
6. **One job per protocol** — Grader scores one axis, AgentFactory builds one agent type
