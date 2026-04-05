# Self-Improving Research Agent

Example application using the `evoagent` library to build a research agent
that improves itself through the Karpathy-style evolution loop.

## Install

```bash
pip install evoagent[all]
pip install deepagents langchain-openai langchain-tavily ddgs
```

## Usage

```python
from evoagent import EvoAgentConfig
from evoagent.core.protocols import AgentFactory
from evoagent.core.types import TaskResult
from evoagent.graders import EfficiencyGrader, MultiJudgeGrader
from evoagent.harness import default_middleware_stack
from evoagent.memory import FileMemoryStore
from evoagent.skills import SkillManager
from evoagent.evolution import analyze_trajectory, classify_trajectory

# 1. Implement AgentFactory for your agent
class ResearchAgentFactory(AgentFactory):
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
middleware = default_middleware_stack(skills_dir=config.skills_path)

# 3. Grade
graders = [
    MultiJudgeGrader(llm=your_llm, name="task_completion"),
    MultiJudgeGrader(llm=your_llm, name="quality"),
    EfficiencyGrader(),
]

# 4. Run the evolution loop
for cycle in range(config.max_cycles):
    for task in tasks:
        result = factory.run(agent, task)
        grades = analyze_trajectory(graders, task, result.output, metrics=...)
        classification, score = classify_trajectory(grades)
```

## Architecture

This example demonstrates the 4-layer self-improvement pattern:

1. **Inner Loop** — Agent runs tasks (search, synthesize, write reports)
2. **Harness** — Middleware catches errors in real-time (self-verification, loop detection)
3. **Sleep-Time** — Cross-run analysis generates meta-instructions between sessions
4. **Outer Loop** — Karpathy-style evolution: propose new prompt, evaluate, keep/reject
