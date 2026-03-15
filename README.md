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
│              EVAL: LangSmith + 3 Graders             │
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

## Quick Start

```bash
# Install
pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Run a single task
make run TASK="What are the latest advances in quantum computing?"

# Run inner-loop evolution (3 cycles)
make evolve CYCLES=3

# View learned artifacts
make skills    # List all skills (success + defensive)
make prompts   # Show prompt version history
make memory    # Browse stored memories
make state     # Show evolution state for outer loop
```

## For the Outer-Loop Coding Agent

If you're a coding agent (Claude Code, Cursor, Codex), read `program.md` for full instructions. The key workflow:

1. Run inner loop: `make evolve CYCLES=5`
2. Read `evolution_state/*.md` to understand what the inner loop learned
3. Make ONE structural change based on the failure analysis
4. Re-run inner loop to validate
5. Keep or discard based on results

## Project Structure

```
src/
├── agent/                    # Agent factory, prompts, sub-agents
│   ├── deep_agent.py         # LangGraph agent with memory + skills
│   ├── prompts.py            # All prompt templates (including failure skill prompt)
│   ├── prompt_store.py       # Versioned prompt persistence
│   └── subagents.py          # Research + synthesis sub-agents
├── evolution/                # Evolution engine
│   ├── orchestrator.py       # Two-speed evolution loop (10 nodes)
│   ├── analyzer.py           # 3-grader trajectory analysis
│   ├── skill_extractor.py    # Extract skills from successes
│   ├── failure_skill_creator.py  # NEW: Create skills from failures
│   ├── prompt_optimizer.py   # Skill-aware prompt optimization
│   ├── evolution_state_bridge.py  # NEW: Bridge inner→outer loop
│   ├── state.py              # LangGraph state schemas
│   └── graders/              # Task completion, efficiency, quality
├── memory/                   # Reflective memory system
│   ├── store.py              # JSON-backed episodic + semantic
│   ├── reflection.py         # Post-run LLM reflection
│   └── compression.py        # Dedup + consolidation
├── skills/                   # SKILL.md management (Anthropic format)
│   └── manager.py            # CRUD with progressive disclosure
├── tools/                    # Agent tools
│   └── search.py             # Tavily web search
├── tracing/                  # LangSmith integration
│   ├── fetcher.py            # Trace fetching
│   └── trajectory.py         # Pydantic trajectory models
├── config/                   # Settings (pydantic-settings)
└── cli/                      # CLI commands

evolution_state/              # Bridge to outer-loop coding agent
├── failures.md               # Deduplicated failure patterns
├── hypotheses.md             # What's been tried + outcomes
├── skills_summary.md         # All skills (success + defensive)
└── plateau_report.md         # Handoff signal to outer loop

program.md                    # Instructions for the outer-loop coding agent
tasks/research_tasks.json     # Evaluation dataset
results.tsv                   # Experiment log (autoresearch style)
```

## How It Differs from Each Parent

| Feature | autoresearch | self_evolving_deep_agents | This repo |
|---------|-------------|--------------------------|-----------|
| Code changes | Yes (outer loop) | No | Yes (outer loop) |
| Prompt optimization | Manual (coding agent) | Auto (metaprompt) | Auto + skill-aware |
| Skill learning | No | From successes only | From successes AND failures |
| Memory | No | Episodic + semantic | Episodic + semantic |
| Failure analysis | No | Grader-based | Grader-based + skill creation |
| State bridge | No | No | Yes (evolution_state/) |
| Plateau detection | No | Yes (5% threshold) | Yes + outer-loop handoff |
| Skill format | N/A | Basic SKILL.md | Anthropic skill-creator format |
