import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.training.lerobot_to_recap import convert_lerobot_dataset_to_recap
from openpi.training.recap_episode_io import load_recap_episodes


def test_convert_lerobot_dataset_to_recap_json(tmp_path):
    dataset_root = tmp_path / "lerobot"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "data/chunk-000").mkdir(parents=True)
    (dataset_root / "videos/chunk-000/observation.images.cam").mkdir(parents=True)
    (dataset_root / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 10.0,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    "action": {"dtype": "float32", "shape": [2]},
                    "observation.state": {"dtype": "float32", "shape": [2]},
                    "observation.images.cam": {"dtype": "video", "shape": [4, 4, 3]},
                },
            }
        ),
        encoding="utf-8",
    )
    (dataset_root / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["move tube"], "length": 2}) + "\n",
        encoding="utf-8",
    )
    (dataset_root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "move tube"}) + "\n",
        encoding="utf-8",
    )
    table = pa.table(
        {
            "action": [[[0.1, 0.2], [0.3, 0.4]][idx] for idx in range(2)],
            "observation.state": [[[1.0, 1.1], [2.0, 2.1]][idx] for idx in range(2)],
            "timestamp": [0.0, 0.1],
            "frame_index": [0, 1],
            "episode_index": [0, 0],
            "task_index": [0, 0],
        }
    )
    pq.write_table(table, dataset_root / "data/chunk-000/episode_000000.parquet")

    output_dir = tmp_path / "recap"
    convert_lerobot_dataset_to_recap(dataset_root, output_dir, default_success=None)

    episodes = load_recap_episodes(output_dir)
    assert len(episodes) == 1
    assert episodes[0].episode_id == "episode_000000"
    assert episodes[0].task == "move tube"
    assert episodes[0].success is False
    assert episodes[0].metadata["success_needs_review"] is True
    assert episodes[0].metadata["video_paths"]["observation.images.cam"].endswith("episode_000000.mp4")
    assert episodes[0].frames[1].t == 1
    assert episodes[0].frames[1].observation["timestamp"] == 0.1
    assert episodes[0].frames[1].observation["frame_index"] == 1
    assert episodes[0].frames[1].action.tolist() == pytest.approx([0.3, 0.4])
    assert episodes[0].frames[1].observation["state"].tolist() == pytest.approx([2.0, 2.1])
