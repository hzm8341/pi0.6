import json

import numpy as np

from openpi.training.recap_episode_io import load_recap_episodes, save_recap_episodes


def test_load_recap_episode_json_converts_frames(tmp_path):
    path = tmp_path / "episode.json"
    path.write_text(
        json.dumps(
            {
                "episode_id": "ep001",
                "task": "insert rack",
                "success": True,
                "max_episode_length": 10,
                "frames": [
                    {
                        "t": 0,
                        "observation": {"state": [0.0, 1.0]},
                        "action": [0.1, 0.2],
                        "is_human_intervention": False,
                    },
                    {
                        "t": 1,
                        "observation": {"state": [1.0, 2.0]},
                        "action": [0.3, 0.4],
                        "is_human_intervention": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    episodes = load_recap_episodes(path)

    assert len(episodes) == 1
    assert episodes[0].episode_id == "ep001"
    assert episodes[0].frames[1].is_human_intervention is True
    np.testing.assert_allclose(episodes[0].frames[0].action, np.array([0.1, 0.2], dtype=np.float32))
    np.testing.assert_allclose(episodes[0].frames[1].observation["state"], np.array([1.0, 2.0], dtype=np.float32))


def test_save_recap_episodes_round_trips_directory(tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "episode_id": "ep002",
                "task": "debug",
                "success": False,
                "frames": [{"t": 0, "observation": {"state": [0.0]}, "action": [0.0]}],
            }
        ),
        encoding="utf-8",
    )
    episodes = load_recap_episodes(input_path)

    save_recap_episodes(episodes, tmp_path / "episodes")
    loaded = load_recap_episodes(tmp_path / "episodes")

    assert [episode.episode_id for episode in loaded] == ["ep002"]
    assert loaded[0].success is False
