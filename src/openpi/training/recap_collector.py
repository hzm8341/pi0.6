from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np


@dataclasses.dataclass(frozen=True)
class ReCAPEpisode:
    observations: Sequence[dict]
    actions: Sequence[np.ndarray]
    rewards: np.ndarray
    success: bool
    is_human_intervention: np.ndarray
    advantage_indicator: np.ndarray | None = None


def assign_advantage_labels(
    episodes: Sequence[ReCAPEpisode],
    *,
    advantages: Sequence[np.ndarray],
    positive_quantile: float,
) -> list[ReCAPEpisode]:
    if len(episodes) != len(advantages):
        raise ValueError(f"Expected one advantage array per episode, got {len(episodes)} episodes and {len(advantages)} arrays.")

    flat_advantages = np.concatenate([np.asarray(adv, dtype=np.float32) for adv in advantages], axis=0)
    threshold = np.quantile(flat_advantages, positive_quantile)

    labeled = []
    for episode, episode_advantages in zip(episodes, advantages, strict=True):
        labels = np.asarray(episode_advantages, dtype=np.float32) >= threshold
        labeled.append(dataclasses.replace(episode, advantage_indicator=labels))
    return labeled
