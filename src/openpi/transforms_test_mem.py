"""Tests for MEM-specific transforms (VideoFrameStack, TokenizeMemory)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from openpi.transforms import VideoFrameStack, TokenizeMemory


class TestVideoFrameStack:
    def test_passthrough_without_history(self):
        data = {"image": {"base_0_rgb": np.zeros((224, 224, 3))}, "state": np.zeros(32)}
        result = VideoFrameStack(num_frames=6)(data)
        assert "image_history" not in result

    def test_valid_history_passthrough(self):
        K = 6
        data = {
            "image": {"base_0_rgb": np.zeros((224, 224, 3))},
            "image_history": {"base_0_rgb": np.zeros((K - 1, 224, 224, 3))},
        }
        result = VideoFrameStack(num_frames=K)(data)
        assert result["image_history"]["base_0_rgb"].shape[0] == K - 1

    def test_wrong_history_length_raises(self):
        K = 6
        data = {
            "image": {"base_0_rgb": np.zeros((224, 224, 3))},
            "image_history": {
                "base_0_rgb": np.zeros((K - 2, 224, 224, 3)),  # wrong: K-2 not K-1
            },
        }
        with pytest.raises(ValueError, match="history has"):
            VideoFrameStack(num_frames=K)(data)


class TestTokenizeMemory:
    @pytest.fixture()
    def tok(self):
        m = MagicMock()
        m.encode.return_value = [10, 20, 30, 40, 50]
        return m

    def test_adds_tokenized_fields(self, tok):
        data = {"language_memory": "I placed the pot in the sink."}
        result = TokenizeMemory(tokenizer=tok, max_len=16)(data)
        assert "tokenized_memory" in result
        assert "tokenized_memory_mask" in result
        assert len(result["tokenized_memory"]) == 16
        assert len(result["tokenized_memory_mask"]) == 16

    def test_padding_correct(self, tok):
        tok.encode.return_value = [1, 2, 3]  # 3 tokens
        data = {"language_memory": "short"}
        result = TokenizeMemory(tokenizer=tok, max_len=8)(data)
        assert result["tokenized_memory"].tolist() == [1, 2, 3, 0, 0, 0, 0, 0]
        assert result["tokenized_memory_mask"].tolist() == [True, True, True] + [False] * 5

    def test_truncation(self, tok):
        tok.encode.return_value = list(range(100))  # 100 tokens
        data = {"language_memory": "very long text"}
        result = TokenizeMemory(tokenizer=tok, max_len=10)(data)
        assert len(result["tokenized_memory"]) == 10
        assert all(result["tokenized_memory_mask"])  # all valid, no padding

    def test_passthrough_without_memory_key(self, tok):
        data = {"state": np.zeros(32)}
        result = TokenizeMemory(tokenizer=tok, max_len=16)(data)
        assert "tokenized_memory" not in result

    def test_passthrough_without_tokenizer(self):
        data = {"language_memory": "some text"}
        result = TokenizeMemory(tokenizer=None, max_len=16)(data)
        assert "tokenized_memory" not in result
