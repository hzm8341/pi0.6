import json

import numpy as np

from openpi.training.recap_dataset_tools import (
    build_split_manifest,
    summarize_episodes,
    trim_tail_static_frames,
)
from openpi.training.recap_episode_io import ReCAPFrame, ReCAPOfflineEpisode


def _episode(episode_id: str, action_values: list[float]) -> ReCAPOfflineEpisode:
    return ReCAPOfflineEpisode(
        episode_id=episode_id,
        task="debug",
        success=True,
        frames=[
            ReCAPFrame(
                t=idx,
                observation={"state": np.array([float(idx)], dtype=np.float32)},
                action=np.array([value], dtype=np.float32),
            )
            for idx, value in enumerate(action_values)
        ],
    )


def test_build_split_manifest_has_disjoint_train_and_eval():
    episodes = [_episode(f"ep{idx}", [0.1, 0.2]) for idx in range(10)]

    manifest = build_split_manifest(episodes, eval_fraction=0.3, seed=7)

    train_ids = {item["episode_id"] for item in manifest["splits"]["train"]}
    eval_ids = {item["episode_id"] for item in manifest["splits"]["eval"]}
    assert len(train_ids) == 7
    assert len(eval_ids) == 3
    assert train_ids.isdisjoint(eval_ids)


def test_trim_tail_static_frames_keeps_minimum_frames():
    episode = _episode("ep001", [0.5, 0.4, 0.0, 0.0, 0.0])

    trimmed = trim_tail_static_frames([episode], action_norm_threshold=0.05, min_frames=3)

    assert len(trimmed[0].frames) == 3
    assert trimmed[0].frames[-1].t == 2


def test_summarize_episodes_reports_action_and_static_stats(tmp_path):
    episodes = [_episode("ep001", [0.0, 0.2, 0.0])]

    summary = summarize_episodes(episodes, static_action_threshold=0.05)
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    assert summary["episode_count"] == 1
    assert summary["total_frames"] == 3
    assert summary["mean_action_norm"] > 0.0
    assert summary["static_frame_ratio"] == 2 / 3
