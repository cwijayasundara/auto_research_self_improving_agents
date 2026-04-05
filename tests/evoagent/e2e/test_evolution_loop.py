"""E2E test for evolution components with a toy agent."""

from evoagent.core.protocols import AgentFactory
from evoagent.core.types import TaskResult, TrajectoryMetrics
from evoagent.evolution.analyzer import analyze_trajectory, classify_trajectory
from evoagent.graders.efficiency import EfficiencyGrader
from evoagent.memory.store import FileMemoryStore
from evoagent.skills.manager import SkillManager


class ToyAgent(AgentFactory):
    """Returns canned responses for testing."""

    def create(self, system_prompt, middleware, **kwargs):
        return system_prompt

    def run(self, agent, task, timeout=60):
        return TaskResult(
            task=task,
            output=f"# Report\n## Summary\nAnalysis of {task}.\n## Finding\nKey finding.\n## Source\n- Source 1\n" + "x" * 500,
            tool_calls=[],
            duration_seconds=10.0,
            status="success",
        )


def test_full_grading_cycle(tmp_path):
    factory = ToyAgent()
    agent = factory.create(system_prompt="You are helpful.", middleware=[])
    result = factory.run(agent, task="Test quantum computing")

    graders = [EfficiencyGrader()]
    grades = analyze_trajectory(
        graders=graders,
        task=result.task,
        output=result.output,
        metrics=TrajectoryMetrics(total_tokens=5000, total_steps=3, latency_seconds=10.0),
    )
    classification, avg_score = classify_trajectory(grades)
    assert classification in ("successful", "partial", "failed")
    assert 0 <= avg_score <= 1


def test_memory_persists_across_cycles(tmp_path):
    store = FileMemoryStore(tmp_path / "memory")
    store.store("episodic", "cycle-1", {"task": "test", "score": 0.7})
    store.store("episodic", "cycle-2", {"task": "test", "score": 0.85})

    all_ep = store.list_all("episodic")
    assert len(all_ep) == 2


def test_skills_created_during_cycle(tmp_path):
    mgr = SkillManager(tmp_path / "skills")
    mgr.create("test-skill", "A test skill", "# Do the thing")
    assert "test-skill" in mgr.discover()
