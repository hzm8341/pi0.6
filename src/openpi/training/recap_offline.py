from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi.training.recap_collector import ReCAPEpisode, assign_advantage_labels
from openpi.training.recap_episode_io import ReCAPOfflineEpisode
from openpi.training.recap_value_proxy import (
    compute_empirical_returns,
    compute_n_step_advantages,
    compute_progress_value_proxy,
    compute_proxy_rewards,
)


@dataclasses.dataclass(frozen=True)
class OfflineReCAPResult:
    records: list[dict[str, Any]]
    fields: dict[str, np.ndarray]


def build_offline_recap_labels(
    episodes: list[ReCAPOfflineEpisode],
    *,
    positive_fraction: float,
    n_step_lookahead: int,
) -> OfflineReCAPResult:
    collector_episodes: list[ReCAPEpisode] = []
    advantages_by_episode: list[np.ndarray] = []
    per_episode_values: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for episode in episodes:
        rewards = compute_proxy_rewards(episode)
        returns = compute_empirical_returns(rewards)
        values = compute_progress_value_proxy(episode)
        advantages = compute_n_step_advantages(rewards, values, n_step_lookahead=n_step_lookahead)
        interventions = np.asarray([frame.is_human_intervention for frame in episode.frames], dtype=bool)

        collector_episodes.append(
            ReCAPEpisode(
                observations=[frame.observation for frame in episode.frames],
                actions=[frame.action for frame in episode.frames],
                rewards=rewards,
                success=episode.success,
                is_human_intervention=interventions,
            )
        )
        advantages_by_episode.append(advantages)
        per_episode_values.append((rewards, returns, values, advantages))

    labeled = assign_advantage_labels(
        collector_episodes,
        advantages=advantages_by_episode,
        positive_fraction=positive_fraction,
    )

    records: list[dict[str, Any]] = []
    advantage_indicator = []
    use_advantage = []
    is_human_intervention = []
    value_target = []
    for episode, labeled_episode, value_tuple in zip(episodes, labeled, per_episode_values, strict=True):
        rewards, returns, values, advantages = value_tuple
        for idx, frame in enumerate(episode.frames):
            records.append(
                {
                    "episode_id": episode.episode_id,
                    "task": episode.task,
                    "t": int(frame.t),
                    "success": episode.success,
                    "reward": float(rewards[idx]),
                    "return": float(returns[idx]),
                    "value": float(values[idx]),
                    "advantage": float(advantages[idx]),
                    "advantage_indicator": bool(labeled_episode.advantage_indicator[idx]),
                    "use_advantage": True,
                    "is_human_intervention": bool(labeled_episode.is_human_intervention[idx]),
                    "label_source": str(labeled_episode.label_source[idx]),
                }
            )
            advantage_indicator.append(bool(labeled_episode.advantage_indicator[idx]))
            use_advantage.append(True)
            is_human_intervention.append(bool(labeled_episode.is_human_intervention[idx]))
            value_target.append(float(returns[idx]))

    return OfflineReCAPResult(
        records=records,
        fields={
            "advantage_indicator": np.asarray(advantage_indicator, dtype=bool),
            "use_advantage": np.asarray(use_advantage, dtype=bool),
            "is_human_intervention": np.asarray(is_human_intervention, dtype=bool),
            # Normalized empirical returns in [-1, 0]: training targets for the value head.
            "value_target": np.asarray(value_target, dtype=np.float32),
        },
    )


def relabel_with_values(
    records: list[dict[str, Any]],
    values: np.ndarray,
    *,
    positive_fraction: float,
    n_step_lookahead: int,
) -> OfflineReCAPResult:
    """Recomputes advantage labels from learned per-frame values.

    `records` are the rows of a previous `recap_labels.jsonl` (same frame order as the training
    dataset and as `values`). Rewards, episode grouping and human-intervention flags are reused;
    only V(o_t) changes, so A_t = sum_{k<N} r_{t+k} + V(o_{t+N}) - V(o_t) can now separate fast from
    slow progress within successful episodes.
    """
    values = np.asarray(values, dtype=np.float32)
    if len(records) != len(values):
        raise ValueError(f"Got {len(records)} label records but {len(values)} values.")

    # Group consecutive frames by episode, preserving order.
    groups: list[list[int]] = []
    for idx, record in enumerate(records):
        if not groups or records[groups[-1][0]]["episode_id"] != record["episode_id"]:
            groups.append([])
        groups[-1].append(idx)
    seen = [records[group[0]]["episode_id"] for group in groups]
    if len(seen) != len(set(seen)):
        raise ValueError("Frames of each episode must be contiguous in the label records.")

    collector_episodes = []
    advantages_by_episode = []
    for group in groups:
        rewards = np.asarray([records[i]["reward"] for i in group], dtype=np.float32)
        advantages_by_episode.append(
            compute_n_step_advantages(rewards, values[group], n_step_lookahead=n_step_lookahead)
        )
        collector_episodes.append(
            ReCAPEpisode(
                observations=[{}] * len(group),
                actions=[np.zeros(0)] * len(group),
                rewards=rewards,
                success=bool(records[group[0]]["success"]),
                is_human_intervention=np.asarray([records[i]["is_human_intervention"] for i in group], dtype=bool),
            )
        )

    labeled = assign_advantage_labels(
        collector_episodes, advantages=advantages_by_episode, positive_fraction=positive_fraction
    )

    new_records: list[dict[str, Any]] = []
    for group, labeled_episode, advantages in zip(groups, labeled, advantages_by_episode, strict=True):
        for local, idx in enumerate(group):
            record = dict(records[idx])
            record.update(
                value=float(values[idx]),
                advantage=float(advantages[local]),
                advantage_indicator=bool(labeled_episode.advantage_indicator[local]),
                label_source=str(labeled_episode.label_source[local]),
                value_source="learned",
            )
            new_records.append(record)

    return OfflineReCAPResult(
        records=new_records,
        fields={
            "advantage_indicator": np.asarray([r["advantage_indicator"] for r in new_records], dtype=bool),
            "use_advantage": np.asarray([r.get("use_advantage", True) for r in new_records], dtype=bool),
            "is_human_intervention": np.asarray([r["is_human_intervention"] for r in new_records], dtype=bool),
            "value_target": np.asarray([r["return"] for r in new_records], dtype=np.float32),
        },
    )


def load_label_records(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_recap_label_outputs(result: OfflineReCAPResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "recap_labels.jsonl"
    labels_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in result.records),
        encoding="utf-8",
    )
    np.savez(output_dir / "lerobot_fields.npz", **result.fields)
