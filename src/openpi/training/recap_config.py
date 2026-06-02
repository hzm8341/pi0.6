from __future__ import annotations

import dataclasses
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class ReCAPIterationConfig:
    collect_episodes: int = 100
    value_train_steps: int = 10_000
    vla_train_steps: int = 30_000
    positive_quantile: float = 0.4


@dataclasses.dataclass(frozen=True)
class ReCAPTrainConfig:
    task_name: str
    demo_dataset_path: str
    output_dir: Path = Path("outputs/recap")
    num_iterations: int = 3
    pretrained_vla_checkpoint: str = ""
    pretrained_vf_checkpoint: str | None = None
    iteration: ReCAPIterationConfig = dataclasses.field(default_factory=ReCAPIterationConfig)
