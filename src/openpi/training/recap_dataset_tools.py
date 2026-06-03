from __future__ import annotations

import dataclasses
import random
from typing import Any

import numpy as np

from openpi.training.recap_episode_io import ReCAPOfflineEpisode


def build_split_manifest(
    episodes: list[ReCAPOfflineEpisode],
    *,
    eval_fraction: float,
    seed: int,
) -> dict[str, Any]:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in (0, 1), got {eval_fraction}.")
    episode_ids = [episode.episode_id for episode in episodes]
    shuffled = episode_ids[:]
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, round(len(shuffled) * eval_fraction))
    eval_ids = set(shuffled[:eval_count])

    def record(episode: ReCAPOfflineEpisode) -> dict[str, Any]:
        return {
            "episode_id": episode.episode_id,
            "task": episode.task,
            "success": episode.success,
            "num_frames": len(episode.frames),
        }

    return {
        "seed": seed,
        "eval_fraction": eval_fraction,
        "splits": {
            "train": [record(episode) for episode in episodes if episode.episode_id not in eval_ids],
            "eval": [record(episode) for episode in episodes if episode.episode_id in eval_ids],
        },
    }


def trim_tail_static_frames(
    episodes: list[ReCAPOfflineEpisode],
    *,
    action_norm_threshold: float,
    min_frames: int,
) -> list[ReCAPOfflineEpisode]:
    if min_frames < 1:
        raise ValueError(f"min_frames must be >= 1, got {min_frames}.")
    trimmed = []
    for episode in episodes:
        keep_length = len(episode.frames)
        while keep_length > min_frames:
            action_norm = float(np.linalg.norm(episode.frames[keep_length - 1].action))
            if action_norm > action_norm_threshold:
                break
            keep_length -= 1
        metadata = dict(episode.metadata)
        metadata["trimmed_tail_frames"] = len(episode.frames) - keep_length
        trimmed.append(dataclasses.replace(episode, frames=episode.frames[:keep_length], metadata=metadata))
    return trimmed


def summarize_episodes(
    episodes: list[ReCAPOfflineEpisode],
    *,
    static_action_threshold: float,
) -> dict[str, Any]:
    action_norms = []
    frame_count = 0
    static_frames = 0
    for episode in episodes:
        for frame in episode.frames:
            action_norm = float(np.linalg.norm(frame.action))
            action_norms.append(action_norm)
            frame_count += 1
            static_frames += int(action_norm <= static_action_threshold)

    return {
        "episode_count": len(episodes),
        "total_frames": frame_count,
        "mean_action_norm": float(np.mean(action_norms)) if action_norms else 0.0,
        "max_action_norm": float(np.max(action_norms)) if action_norms else 0.0,
        "static_frame_ratio": float(static_frames / frame_count) if frame_count else 0.0,
    }
