"""Tests for π0.6-MEM Observation extensions (model.py).

Verifies backward compatibility with π0.5 and correct handling of new fields.
"""

from __future__ import annotations

import numpy as np
import pytest

from openpi.models.model import Observation, preprocess_observation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pi05_dict(batch: int = 2) -> dict:
    """Minimal data dict in the original π0.5 format (no MEM fields)."""
    return {
        "image": {
            cam: np.random.rand(batch, 224, 224, 3).astype(np.float32)
            for cam in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        "image_mask": {
            cam: np.ones(batch, dtype=bool)
            for cam in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        "state": np.random.rand(batch, 32).astype(np.float32),
        "tokenized_prompt": np.zeros((batch, 48), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((batch, 48), dtype=bool),
    }


# ---------------------------------------------------------------------------
# Backward-compatibility tests
# ---------------------------------------------------------------------------

class TestObservationBackwardCompat:
    def test_from_dict_no_mem_fields(self):
        """π0.5-format dict → MEM fields must be None."""
        obs = Observation.from_dict(_make_pi05_dict())
        assert obs.image_history is None
        assert obs.image_history_masks is None
        assert obs.state_history is None
        assert obs.tokenized_memory is None
        assert obs.tokenized_memory_mask is None

    def test_core_fields_unchanged(self):
        """images / state / prompt still populated correctly."""
        data = _make_pi05_dict(batch=3)
        obs = Observation.from_dict(data)
        assert set(obs.images.keys()) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
        assert obs.state.shape == (3, 32)
        assert obs.tokenized_prompt.shape == (3, 48)

    def test_preprocess_no_mem_fields(self):
        """preprocess_observation must not crash on π0.5 observations."""
        obs = Observation.from_dict(_make_pi05_dict())
        processed = preprocess_observation(None, obs, train=False)
        # MEM fields still None
        assert processed.image_history is None
        assert processed.state_history is None
        assert processed.tokenized_memory is None


# ---------------------------------------------------------------------------
# MEM field parsing
# ---------------------------------------------------------------------------

class TestObservationMEMFields:
    def test_image_history_parsed(self):
        K = 5
        data = _make_pi05_dict(batch=2)
        data["image_history"] = {
            "base_0_rgb": np.random.rand(2, K, 224, 224, 3).astype(np.float32),
        }
        obs = Observation.from_dict(data)
        assert obs.image_history is not None
        assert obs.image_history["base_0_rgb"].shape == (2, K, 224, 224, 3)

    def test_image_history_uint8_normalised(self):
        """uint8 history frames must be normalised to [-1, 1]."""
        K = 3
        data = _make_pi05_dict(batch=1)
        data["image_history"] = {
            "base_0_rgb": (np.random.rand(1, K, 224, 224, 3) * 255).astype(np.uint8),
        }
        obs = Observation.from_dict(data)
        hist = obs.image_history["base_0_rgb"]
        assert hist.dtype == np.float32
        assert float(hist.min()) >= -1.0 - 1e-5
        assert float(hist.max()) <= 1.0 + 1e-5

    def test_state_history_parsed(self):
        K = 6
        data = _make_pi05_dict(batch=2)
        data["state_history"] = np.random.rand(2, K, 32).astype(np.float32)
        obs = Observation.from_dict(data)
        assert obs.state_history is not None
        assert obs.state_history.shape == (2, K, 32)

    def test_tokenized_memory_parsed(self):
        data = _make_pi05_dict(batch=2)
        data["tokenized_memory"] = np.zeros((2, 256), dtype=np.int32)
        data["tokenized_memory_mask"] = np.ones((2, 256), dtype=bool)
        obs = Observation.from_dict(data)
        assert obs.tokenized_memory is not None
        assert obs.tokenized_memory.shape == (2, 256)
        assert obs.tokenized_memory_mask is not None

    def test_preprocess_passes_through_mem_fields(self):
        K = 3
        data = _make_pi05_dict(batch=2)
        data["image_history"] = {
            "base_0_rgb": np.zeros((2, K, 224, 224, 3), dtype=np.float32),
        }
        data["state_history"] = np.zeros((2, K, 32), dtype=np.float32)
        data["tokenized_memory"] = np.zeros((2, 256), dtype=np.int32)
        data["tokenized_memory_mask"] = np.ones((2, 256), dtype=bool)
        obs = Observation.from_dict(data)
        processed = preprocess_observation(None, obs, train=False)
        assert processed.image_history is not None
        assert processed.state_history is not None
        assert processed.tokenized_memory is not None
