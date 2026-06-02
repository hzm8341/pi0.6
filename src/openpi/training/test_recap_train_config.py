from pathlib import Path

from openpi.training.recap_config import ReCAPTrainConfig


def test_recap_train_config_defaults():
    config = ReCAPTrainConfig(task_name="debug", demo_dataset_path="demo")
    assert config.output_dir == Path("outputs/recap")
    assert config.num_iterations == 3
    assert config.demo_dataset_path == "demo"
