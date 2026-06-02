from __future__ import annotations

import numpy as np

from openpi.training.recap_episode_io import ReCAPOfflineEpisode


def compute_proxy_rewards(
    episode: ReCAPOfflineEpisode,
    *,
    step_penalty: float = -1.0,
    fail_penalty: float = -100.0,
) -> np.ndarray:
    episode_length = len(episode.frames)
    max_episode_length = episode.max_episode_length or episode_length
    rewards = np.full(episode_length, step_penalty, dtype=np.float32)
    rewards[-1] = 0.0 if episode.success else fail_penalty
    scale = abs(float(max_episode_length) * step_penalty)
    return np.clip(rewards / scale, -1.0, 0.0)


def compute_progress_value_proxy(episode: ReCAPOfflineEpisode) -> np.ndarray:
    episode_length = len(episode.frames)
    max_episode_length = float(episode.max_episode_length or episode_length)
    if not episode.success:
        return np.full(episode_length, -1.0, dtype=np.float32)

    remaining_steps = np.maximum(episode_length - 1 - np.arange(episode_length), 0)
    return np.asarray(-remaining_steps / max_episode_length, dtype=np.float32)


def compute_empirical_returns(rewards: np.ndarray) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = np.float32(0.0)
    for idx in range(len(rewards) - 1, -1, -1):
        running = np.float32(running + rewards[idx])
        returns[idx] = running
    return np.clip(returns, -1.0, 0.0)


def compute_n_step_reward_sums(rewards: np.ndarray, *, n_step_lookahead: int) -> np.ndarray:
    if n_step_lookahead < 1:
        raise ValueError(f"n_step_lookahead must be >= 1, got {n_step_lookahead}.")
    rewards = np.asarray(rewards, dtype=np.float32)
    reward_sums = np.zeros_like(rewards, dtype=np.float32)
    for idx in range(len(rewards)):
        reward_sums[idx] = np.sum(rewards[idx : min(idx + n_step_lookahead, len(rewards))], dtype=np.float32)
    return reward_sums


def compute_n_step_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    *,
    n_step_lookahead: int,
) -> np.ndarray:
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if rewards.shape != values.shape:
        raise ValueError(f"Expected rewards and values to have the same shape, got {rewards.shape} and {values.shape}.")

    reward_sums = compute_n_step_reward_sums(rewards, n_step_lookahead=n_step_lookahead)
    next_indices = np.minimum(np.arange(len(values)) + n_step_lookahead, len(values) - 1)
    return np.asarray(reward_sums + values[next_indices] - values, dtype=np.float32)
