from unittest.mock import MagicMock, patch
from pathlib import Path
from src.evolution.run_log import RunLogEntry


class TestDaemonHarnessOptimization:
    def test_evolution_cycle_calls_harness_optimizer(self):
        from src.evolution.daemon import _run_evolution_cycle

        entry = RunLogEntry(
            run_id="r1",
            task="test",
            output="report",
            classification="partial",
            average_score=0.72,
            grader_results=[
                {
                    "name": "task_completion",
                    "score": 0.4,
                    "passed": False,
                    "reasoning": "missed",
                }
            ],
            prompt_version=7,
            harness_config_version=1,
        )

        with (
            patch("src.evolution.daemon.create_llm") as mock_llm_factory,
            patch("src.evolution.daemon.optimize_prompt") as mock_opt,
            patch(
                "src.evolution.daemon.extract_skills_from_batch"
            ) as mock_skills,
            patch("src.evolution.daemon.deduplicate_semantic") as mock_dedup,
            patch("src.evolution.daemon.optimize_harness") as mock_harness,
            patch(
                "src.evolution.daemon.HarnessConfigStore"
            ) as mock_store_cls,
            patch("src.evolution.daemon.PromptStore"),
            patch("src.evolution.daemon.FileMemoryStore"),
            patch("src.evolution.daemon.SkillManager"),
        ):
            mock_llm_factory.return_value = MagicMock()
            mock_opt.return_value = 8
            mock_skills.return_value = []
            mock_dedup.return_value = 0
            mock_store_instance = MagicMock()
            mock_store_instance.get_latest_version.return_value = 1
            mock_store_cls.return_value = mock_store_instance
            mock_harness.return_value = 2

            settings = MagicMock()
            settings.skills_path = Path("/tmp/skills")
            settings.prompts_path = Path("/tmp/prompts")
            settings.memory_path = Path("/tmp/memory")
            settings.harness_config_path = Path("/tmp/harness")
            settings.compression_similarity_threshold = 0.7

            result = _run_evolution_cycle(settings, [entry])
            mock_harness.assert_called_once()
            assert result["harness_changed"] is True
