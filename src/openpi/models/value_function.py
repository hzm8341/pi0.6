from __future__ import annotations

import numpy as np

try:
    import flax.nnx as nnx
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal utility environments.
    nnx = None


class DistributionalValueHead(nnx.Module if nnx is not None else object):
    def __init__(self, hidden_dim: int, num_bins: int, *, rngs: nnx.Rngs):
        if nnx is None:
            raise ModuleNotFoundError("flax is required to construct DistributionalValueHead.")
        self.head = nnx.Linear(hidden_dim, num_bins, rngs=rngs)

    def __call__(self, x):
        return self.head(x)


def compute_episode_rewards(
    *,
    success: bool,
    episode_length: int,
    max_episode_length: int,
    step_penalty: float = -1.0,
    fail_penalty: float = -100.0,
) -> np.ndarray:
    rewards = np.full(episode_length, step_penalty, dtype=np.float32)
    rewards[-1] = 0.0 if success else fail_penalty
    scale = abs(max_episode_length * step_penalty)
    return np.clip(rewards / scale, -1.0, 0.0)


def compute_empirical_returns(rewards: np.ndarray) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = np.float32(0.0)
    for idx in range(len(rewards) - 1, -1, -1):
        running = np.float32(running + rewards[idx])
        returns[idx] = running
    return np.clip(returns, -1.0, 0.0)


def return_to_bin(values: np.ndarray, *, num_bins: int, value_min: float, value_max: float) -> np.ndarray:
    clipped = np.clip(values, value_min, value_max)
    scaled = (clipped - value_min) / (value_max - value_min)
    return np.rint(scaled * (num_bins - 1)).astype(np.int32)


def bin_to_return(bins: np.ndarray, *, num_bins: int, value_min: float, value_max: float) -> np.ndarray:
    scaled = bins.astype(np.float32) / float(num_bins - 1)
    return scaled * (value_max - value_min) + value_min


def compute_advantage(rewards_sum: np.ndarray, v_current: np.ndarray, v_next: np.ndarray) -> np.ndarray:
    return rewards_sum + v_next - v_current
