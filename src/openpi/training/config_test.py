from openpi.training import config as _config


def test_scene1_lora_config_is_registered_and_local():
    cfg = _config.get_config("pi05_scene1_right8_chest_rightwrist_lora")

    assert cfg.model.action_dim == 8
    assert cfg.model.action_horizon == 16
    assert cfg.model.pi05 is True
    assert cfg.data.repo_id == "scene1_right8_chest_rightwrist"
    assert cfg.data.assets.asset_id == "scene1_right8_chest_rightwrist"
    assert cfg.data.assets.assets_dir == "./assets/pi05_scene1_right8_chest_wrist_finetune"
    assert cfg.weight_loader.params_path == "./openpi-assets/checkpoints/pi05_base/params"
    assert cfg.freeze_filter is not None
