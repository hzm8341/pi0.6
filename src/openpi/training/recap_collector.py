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
    label_source: np.ndarray | None = None


def assign_advantage_labels(
    episodes: Sequence[ReCAPEpisode],
    *,
    advantages: Sequence[np.ndarray],
    positive_fraction: float,
) -> list[ReCAPEpisode]:
    if not 0.0 < positive_fraction <= 1.0:
        raise ValueError(f"positive_fraction must be in (0, 1], got {positive_fraction}.")
    if len(episodes) != len(advantages):
        raise ValueError(
            f"Expected one advantage array per episode, got {len(episodes)} episodes and {len(advantages)} arrays."
        )

    flat_advantages = np.concatenate([np.asarray(adv, dtype=np.float32) for adv in advantages], axis=0)
    threshold = np.quantile(flat_advantages, 1.0 - positive_fraction)

    labeled = []
    for episode, episode_advantages in zip(episodes, advantages, strict=True):
        episode_advantages = np.asarray(episode_advantages, dtype=np.float32)
        human_interventions = np.asarray(episode.is_human_intervention, dtype=bool)
        if episode_advantages.shape != human_interventions.shape:
            raise ValueError(
                "Expected advantages and is_human_intervention to have the same shape, got "
                f"{episode_advantages.shape} and {human_interventions.shape}."
            )

        value_positive = episode_advantages >= threshold
        labels = np.logical_or(value_positive, human_interventions)
        label_source = np.full(labels.shape, "negative", dtype=object)
        label_source[value_positive] = "value"
        label_source[human_interventions] = "human"
        labeled.append(dataclasses.replace(episode, advantage_indicator=labels, label_source=label_source))
    return labeled
