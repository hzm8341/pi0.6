import numpy as np
import pytest

from openpi import transforms as _transforms
from openpi.models import pi0_config
from openpi.training import data_loader as _data_loader
from openpi.training.recap_episode_io import ReCAPFrame, ReCAPOfflineEpisode
from openpi.training.recap_offline import build_offline_recap_labels, relabel_with_values


def _episode(episode_id: str, length: int, *, success: bool, intervention_at: int | None = None):
    return ReCAPOfflineEpisode(
        episode_id=episode_id,
        task="insert tube",
        success=success,
        max_episode_length=100,
        frames=[
            ReCAPFrame(
                t=t,
                observation={},
                action=np.ones(2, dtype=np.float32),
                is_human_intervention=t == intervention_at,
            )
            for t in range(length)
        ],
    )


def test_offline_labels_export_value_targets():
    result = build_offline_recap_labels(
        [_episode("a", 5, success=True), _episode("b", 4, success=False)], positive_fraction=0.5, n_step_lookahead=2
    )
    targets = result.fields["value_target"]
    assert targets.dtype == np.float32
    assert targets.shape == (9,)
    assert ((targets >= -1.0) & (targets <= 0.0)).all()
    # Successful episode: return rises to 0 at the end.
    assert targets[4] == 0.0
    np.testing.assert_allclose(targets, [r["return"] for r in result.records])


def test_relabel_with_learned_values_separates_progress_within_success():
    # One successful episode: frames 0-9 make fast progress, frames 10-19 stall.
    episode = _episode("a", 20, success=True)
    base = build_offline_recap_labels([episode], positive_fraction=0.5, n_step_lookahead=2)
    # The proxy assigns identical advantages to every frame of a successful episode.
    assert len({round(r["advantage"], 6) for r in base.records}) == 1

    values = np.concatenate([np.linspace(-0.9, -0.3, 10), np.full(10, -0.3)]).astype(np.float32)
    values[-1] = 0.0
    result = relabel_with_values(base.records, values, positive_fraction=0.4, n_step_lookahead=2)

    labels = result.fields["advantage_indicator"]
    assert labels[:8].mean() > labels[10:18].mean()
    assert all(r["value_source"] == "learned" for r in result.records)
    np.testing.assert_allclose(result.fields["value_target"], base.fields["value_target"])


def test_relabel_keeps_human_interventions_positive():
    base = build_offline_recap_labels(
        [_episode("a", 6, success=False, intervention_at=3)], positive_fraction=0.5, n_step_lookahead=1
    )
    values = np.linspace(-0.2, -0.9, 6).astype(np.float32)  # Value drops everywhere → negative advantages.
    result = relabel_with_values(base.records, values, positive_fraction=0.1, n_step_lookahead=1)
    assert result.fields["advantage_indicator"][3]
    assert result.records[3]["label_source"] == "human"


def test_relabel_rejects_length_mismatch_and_non_contiguous_episodes():
    base = build_offline_recap_labels(
        [_episode("a", 2, success=True), _episode("b", 2, success=True)], positive_fraction=0.5, n_step_lookahead=1
    )
    with pytest.raises(ValueError, match="values"):
        relabel_with_values(base.records, np.zeros(3), positive_fraction=0.5, n_step_lookahead=1)
    shuffled = [base.records[i] for i in (0, 2, 1, 3)]
    with pytest.raises(ValueError, match="contiguous"):
        relabel_with_values(shuffled, np.zeros(4), positive_fraction=0.5, n_step_lookahead=1)


def test_preserve_fields_survives_repack():
    repack = _transforms.RepackTransform({"state": "observation/state"})
    transform = _transforms.PreserveFields(("advantage_indicator", "value_target"), [repack])
    out = transform(
        {
            "observation": {"state": np.zeros(2)},
            "advantage_indicator": np.asarray(True),
            "value_target": np.asarray(-0.5, dtype=np.float32),
            "unrelated": 1,
        }
    )
    assert set(out) == {"state", "advantage_indicator", "value_target"}


def test_recap_fields_dataset_loads_optional_value_target(tmp_path):
    dataset = _data_loader.FakeDataset(pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48), 2)
    path = tmp_path / "lerobot_fields.npz"
    np.savez(
        path,
        advantage_indicator=np.array([True, False]),
        use_advantage=np.array([True, True]),
        is_human_intervention=np.array([False, False]),
        value_target=np.array([-0.25, -1.0], dtype=np.float32),
    )
    merged = _data_loader.ReCAPFieldsDataset(dataset, path)
    assert merged[1]["value_target"].dtype == np.float32
    assert float(merged[1]["value_target"]) == -1.0
    assert merged[0]["advantage_indicator"].dtype == np.bool_


def test_separated_rewards_keep_failure_penalty():
    from openpi.training.recap_episode_io import _episode_from_dict
    from openpi.training.recap_value_proxy import compute_empirical_returns
    from openpi.training.recap_value_proxy import compute_proxy_rewards

    def episode(length, success):
        frames = [{"t": t, "action": [0.0]} for t in range(length)]
        return _episode_from_dict(
            {"episode_id": "e", "task": "x", "success": success, "max_episode_length": 10, "frames": frames}
        )

    fail = compute_empirical_returns(compute_proxy_rewards(episode(10, success=False), fail_penalty_steps=10))
    slow = compute_empirical_returns(compute_proxy_rewards(episode(10, success=True), fail_penalty_steps=10))
    fast = compute_empirical_returns(compute_proxy_rewards(episode(4, success=True), fail_penalty_steps=10))
    assert fail[0] == pytest.approx(-1.0)
    assert slow[0] == pytest.approx(-0.45)
    assert fast[0] == pytest.approx(-0.15)
    assert fail[-1] < -0.5 < slow[0]
