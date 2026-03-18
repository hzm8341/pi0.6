"""Tests for memory_manager.py and high_level_policy.py."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from openpi.models.memory_manager import (
    MemoryDataGenerator,
    MemoryGenerationConfig,
    MemoryLabel,
)
from openpi.models.high_level_policy import HighLevelPolicy, HighLevelPolicyConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_llm():
    llm = MagicMock()
    llm.generate.return_value = "I placed the pot in the sink and grabbed the potatoes."
    return llm


@pytest.fixture()
def generator(mock_llm):
    return MemoryDataGenerator(
        llm_client=mock_llm,
        config=MemoryGenerationConfig(max_memory_length=256),
    )


@pytest.fixture()
def mock_vlm():
    """Returns a valid JSON string with subtask + updated_memory."""
    def _vlm(image, prompt, max_tokens=512):
        return json.dumps({
            "subtask": "Pick up the bowl",
            "updated_memory": "I cleared the counter and gathered ingredients.",
        })
    return _vlm


@pytest.fixture()
def hl_policy(mock_vlm):
    tok = MagicMock()
    tok.encode.return_value = list(range(10))
    return HighLevelPolicy(
        vlm_inference_fn=mock_vlm,
        tokenizer=tok,
        config=HighLevelPolicyConfig(
            subtask_trigger_steps=5,
            max_memory_tokens=32,
        ),
    )


# ---------------------------------------------------------------------------
# MemoryDataGenerator
# ---------------------------------------------------------------------------

class TestMemoryDataGenerator:
    def test_label_count_matches_subtask_count(self, generator):
        subtasks = [
            {"instruction": "Step A", "success": True},
            {"instruction": "Step B", "success": True},
            {"instruction": "Step C", "success": False},
        ]
        labels = generator.generate_labels_for_episode("ep1", subtasks)
        assert len(labels) == 3

    def test_first_step_memory_before_is_empty(self, generator):
        subtasks = [{"instruction": "Start", "success": True}]
        labels = generator.generate_labels_for_episode("ep2", subtasks)
        assert labels[0].memory_before == ""

    def test_failed_step_does_not_advance_memory(self, generator):
        subtasks = [
            {"instruction": "A", "success": True},
            {"instruction": "B", "success": False},
            {"instruction": "C", "success": True},
        ]
        labels = generator.generate_labels_for_episode("ep3", subtasks)
        # Step 2 (index=2) memory_before must equal step 1 (index=1) memory_before
        # because step 1 failed and should not update memory.
        assert labels[2].memory_before == labels[1].memory_before, (
            "Failed subtask should not update language memory"
        )

    def test_episode_id_propagated(self, generator):
        labels = generator.generate_labels_for_episode(
            "my_episode", [{"instruction": "X", "success": True}]
        )
        assert labels[0].episode_id == "my_episode"

    def test_save_and_load_roundtrip(self, generator):
        labels = [
            MemoryLabel("ep", i, f"sub{i}", True, f"before{i}", f"after{i}")
            for i in range(4)
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            generator.save_labels(labels, path)
            loaded = MemoryDataGenerator.load_labels(path)
            assert len(loaded) == len(labels)
            for orig, got in zip(labels, loaded):
                assert orig.episode_id == got.episode_id
                assert orig.memory_before == got.memory_before
                assert orig.memory_after == got.memory_after
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# HighLevelPolicy
# ---------------------------------------------------------------------------

class TestHighLevelPolicy:
    def test_reset_clears_state(self, hl_policy):
        hl_policy._language_memory = "old memory"
        hl_policy._episode_step = 99
        hl_policy.reset("Clean the kitchen")
        assert hl_policy.language_memory == ""
        assert hl_policy._episode_step == 0
        assert hl_policy._task_goal == "Clean the kitchen"

    def test_update_returns_subtask_and_memory(self, hl_policy):
        hl_policy.reset("Task")
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        subtask, memory = hl_policy.update(img, subtask_success=True)
        assert isinstance(subtask, str) and len(subtask) > 0
        assert isinstance(memory, str)

    def test_failed_update_does_not_change_memory(self, hl_policy):
        hl_policy.reset("Task")
        hl_policy._language_memory = "original memory"
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        _, memory = hl_policy.update(img, subtask_success=False)
        assert memory == "original memory", (
            "Memory must not be updated when subtask_success=False"
        )

    def test_should_update_step_trigger(self, hl_policy):
        hl_policy.reset("T")
        steps = hl_policy.config.subtask_trigger_steps  # 5
        results = [hl_policy.should_update() for _ in range(steps)]
        assert results[-1] is True
        assert results[0] is False

    def test_should_update_completion_trigger(self, hl_policy):
        hl_policy.reset("T")
        assert hl_policy.should_update(subtask_completed=True) is True

    def test_tokenize_memory_length(self, hl_policy):
        ids, mask = hl_policy.tokenize_memory("some memory text")
        assert len(ids) == hl_policy.config.max_memory_tokens
        assert len(mask) == hl_policy.config.max_memory_tokens

    def test_tokenize_empty_memory(self, hl_policy):
        ids, mask = hl_policy.tokenize_memory("")
        assert all(i == 0 for i in ids)
        assert not any(mask)

    def test_tokenize_truncates_long_text(self, hl_policy):
        # tokenizer returns 50 tokens, max_memory_tokens=32 → must truncate
        hl_policy._tokenizer.encode.return_value = list(range(50))
        ids, mask = hl_policy.tokenize_memory("long text")
        assert len(ids) == hl_policy.config.max_memory_tokens
        assert all(mask)  # all slots filled (no padding)

    def test_bad_vlm_json_falls_back_gracefully(self):
        """Non-JSON VLM output must not crash; policy retains previous state."""
        def bad_vlm(image, prompt, max_tokens=512):
            return "not json at all"

        tok = MagicMock()
        tok.encode.return_value = [1, 2]
        policy = HighLevelPolicy(bad_vlm, tok)
        policy.reset("T")
        policy._language_memory = "safe memory"
        policy._current_subtask = "safe subtask"

        img = np.zeros((224, 224, 3), dtype=np.uint8)
        subtask, memory = policy.update(img, subtask_success=True)
        # Falls back to previous values
        assert subtask == "safe subtask"
        assert memory == "safe memory"
