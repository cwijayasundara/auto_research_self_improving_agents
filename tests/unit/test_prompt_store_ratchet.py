"""Tests for PromptStore's ratchet semantics.

These pin the Karpathy-ratchet design:
1. The active version is always the LATEST version, regardless of score
2. ``promotion_score`` is frozen at promotion time and never drifts
3. ``update_score`` only mutates the running ``score``, never ``promotion_score``
4. ``add_version(promotion_score=...)`` allows the optimizer to inherit
   a parent's frozen baseline so each new champion has a stable floor
"""

import json
from pathlib import Path

from src.agent.prompt_store import PromptStore


class TestPromptStoreRatchet:
    def test_get_current_returns_latest_ignoring_scores(self, tmp_path: Path):
        """The active prompt is the LATEST version, even if an older
        version has a higher running score. This is the ratchet — once
        promoted, a champion stays active until it's explicitly replaced
        by a newer promotion, never silently rolled back by score drift."""
        store = PromptStore(tmp_path / "prompts")
        store.add_version("v1 prompt", score=0.5)
        store.add_version("v2 prompt", score=0.9)
        store.add_version("v3 prompt", score=0.4)  # latest, even with low score

        assert store.get_current_prompt() == "v3 prompt"

    def test_get_current_returns_latest_when_unscored(self, tmp_path: Path):
        store = PromptStore(tmp_path / "prompts")
        store.add_version("v1 prompt")
        store.add_version("v2 prompt")
        assert store.get_current_prompt() == "v2 prompt"

    def test_get_current_empty_store(self, tmp_path: Path):
        store = PromptStore(tmp_path / "prompts")
        assert store.get_current_prompt() == ""

    def test_promotion_score_frozen_under_observed_drift(self, tmp_path: Path):
        """The frozen ``promotion_score`` must NOT change when
        ``update_score`` runs, even after many bad observations."""
        store = PromptStore(tmp_path / "prompts")
        store.add_version("baseline", score=0.8)

        # Hammer it with bad observed scores — exactly the v12 scenario
        # that caused the recent score decay (0.8043 → 0.6442 over 4 runs).
        for bad in [0.3, 0.2, 0.4, 0.25, 0.35]:
            store.update_score(1, bad)

        data = json.loads((tmp_path / "prompts" / "v0001.json").read_text())
        assert data["promotion_score"] == 0.8, (
            "promotion_score must remain 0.8 even after observed drift"
        )
        # Running score should have drifted
        assert data["score"] < 0.6
        # Active version is still v1 (only version)
        v = store.get_version(1)
        assert v is not None
        assert v.promotion_score == 0.8
        assert v.score is not None and v.score < 0.6

    def test_explicit_promotion_score_overrides_score(self, tmp_path: Path):
        """When the optimizer wants a new champion to inherit a baseline
        different from the current batch score, the explicit
        ``promotion_score`` arg wins."""
        store = PromptStore(tmp_path / "prompts")
        store.add_version("baseline", score=0.8)
        # Optimizer promotes v2 with batch score 0.5 but inherits parent's 0.8
        store.add_version("evolved", score=0.5, promotion_score=0.8)

        v2 = json.loads((tmp_path / "prompts" / "v0002.json").read_text())
        assert v2["promotion_score"] == 0.8
        assert v2["score"] == 0.5

    def test_old_files_without_promotion_score_load_safely(self, tmp_path: Path):
        """Backward compat: existing JSON files written before the
        promotion_score field existed must load without error and yield
        promotion_score=None."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir(parents=True)
        # Hand-write an old-format file (no promotion_score key)
        (prompts_dir / "v0001.json").write_text(json.dumps({
            "version": 1,
            "prompt": "legacy prompt",
            "score": 0.7,
            "score_count": 5,
            "timestamp": "2026-04-07T00:00:00+00:00",
            "parent_version": None,
            "feedback_summary": "",
        }))

        store = PromptStore(prompts_dir)
        v = store.get_version(1)
        assert v is not None
        assert v.prompt == "legacy prompt"
        assert v.score == 0.7
        assert v.promotion_score is None  # not present in old file
        # Active selection still works
        assert store.get_current_prompt() == "legacy prompt"

    def test_ratchet_survives_bug_that_drags_score_down(self, tmp_path: Path):
        """Regression scenario: a bug like extract_output's used to feed
        zero-score runs into the champion. Verify that even N rounds of
        zeroes do NOT change which version is active."""
        store = PromptStore(tmp_path / "prompts")
        store.add_version("good champion", score=0.8)

        # Simulate the bug: 10 bad runs in a row
        for _ in range(10):
            store.update_score(1, 0.0)

        # Current prompt is unchanged
        assert store.get_current_prompt() == "good champion"
        # promotion_score is still 0.8
        v = store.get_version(1)
        assert v is not None and v.promotion_score == 0.8
        # Observed score has crashed (which is the optimizer's signal to act)
        assert v.score is not None and v.score < 0.2
