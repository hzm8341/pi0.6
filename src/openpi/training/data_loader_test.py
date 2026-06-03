import dataclasses

import jax
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_recap_fields_dataset_merges_sidecar_fields(tmp_path):
    model_config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(model_config, 3)
    fields_path = tmp_path / "lerobot_fields.npz"
    np.savez(
        fields_path,
        advantage_indicator=np.array([True, False, True]),
        use_advantage=np.array([True, True, False]),
        is_human_intervention=np.array([False, True, False]),
    )

    merged = _data_loader.ReCAPFieldsDataset(dataset, fields_path)

    assert merged[0]["advantage_indicator"].item() is True
    assert merged[1]["use_advantage"].item() is True
    assert merged[1]["is_human_intervention"].item() is True
    assert merged[2]["use_advantage"].item() is False


def test_recap_fields_dataset_rejects_length_mismatch(tmp_path):
    model_config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(model_config, 3)
    fields_path = tmp_path / "lerobot_fields.npz"
    np.savez(
        fields_path,
        advantage_indicator=np.array([True, False]),
        use_advantage=np.array([True, True]),
        is_human_intervention=np.array([False, True]),
    )

    try:
        _data_loader.ReCAPFieldsDataset(dataset, fields_path)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected length mismatch to raise ValueError.")


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
