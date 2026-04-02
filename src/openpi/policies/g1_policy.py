"""
G1 Humanoid Robot Policy Transforms for PhysicalAI-Robotics-GR00T-Teleop-G1 dataset.

Dataset: g1-pick-apple
- State/Action: 43 joints (legs, waist, arms, hands)
- Camera: ego_view (480x640x3)
- Tasks: pick up fruits and place on plate
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_g1_example() -> dict:
    """Creates a random input example for the G1 policy."""
    return {
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/state": np.random.rand(43).astype(np.float32),
        "prompt": "Pick up the red apple and place it on the plate",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G1Inputs(transforms.DataTransformFn):
    """Convert G1 dataset inputs to the expected model format."""

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        ego_image = _parse_image(data["observation/image"])

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": ego_image,
                # G1 has only one ego-view camera; pad with zeros for unused slots.
                "left_wrist_0_rgb": np.zeros_like(ego_image),
                "right_wrist_0_rgb": np.zeros_like(ego_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class G1Outputs(transforms.DataTransformFn):
    """Convert model outputs back to G1 action format."""

    action_dim: int = 43

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
