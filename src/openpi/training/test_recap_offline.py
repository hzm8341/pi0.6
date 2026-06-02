import json

import numpy as np

from openpi.training.recap_episode_io import ReCAPFrame, ReCAPOfflineEpisode
from openpi.training.recap_offline import build_offline_recap_labels, write_recap_label_outputs
from openpi.training.recap_value_proxy import compute_n_step_advantages


def test_compute_n_step_advantages_uses_reward_sum_and_next_value():
    rewards = np.array([-0.1, -0.1, 0.0], dtype=np.float32)
    values = np.array([-0.5, -0.2, 0.0], dtype=np.float32)

    advantages = compute_n_step_advantages(rewards, values, n_step_lookahead=2)

    np.testing.assert_allclose(advantages, np.array([0.3, 0.1, 0.0], dtype=np.float32), atol=1e-6)


def test_build_offline_recap_labels_marks_human_positive_and_exports_fields(tmp_path):
    episode = ReCAPOfflineEpisode(
        episode_id="ep001",
        task="insert rack",
        success=True,
        max_episode_length=10,
        frames=[
            ReCAPFrame(t=0, observation={"state": np.array([0.0], dtype=np.float32)}, action=np.array([0.0])),
            ReCAPFrame(
                t=1,
                observation={"state": np.array([1.0], dtype=np.float32)},
                action=np.array([1.0]),
                is_human_intervention=True,
            ),
            ReCAPFrame(t=2, observation={"state": np.array([2.0], dtype=np.float32)}, action=np.array([2.0])),
        ],
    )

    result = build_offline_recap_labels([episode], positive_fraction=0.5, n_step_lookahead=1)
    write_recap_label_outputs(result, tmp_path)

    records = [json.loads(line) for line in (tmp_path / "recap_labels.jsonl").read_text(encoding="utf-8").splitlines()]
    fields = np.load(tmp_path / "lerobot_fields.npz")

    assert [record["episode_id"] for record in records] == ["ep001", "ep001", "ep001"]
    assert records[1]["is_human_intervention"] is True
    assert records[1]["advantage_indicator"] is True
    assert records[1]["label_source"] == "human"
    assert fields["advantage_indicator"].dtype == np.bool_
    assert fields["use_advantage"].tolist() == [True, True, True]
    assert fields["is_human_intervention"].tolist() == [False, True, False]
