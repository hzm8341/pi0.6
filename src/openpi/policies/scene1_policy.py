import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_scene1_example() -> dict:
    """Creates a random input example for the Scene 1 policy."""
    return {
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/state": np.random.rand(8).astype(np.float32),
        "prompt": "Right gripper picks up a test tube from the top source rack and moves it to the bottom destination rack.",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Scene1Inputs(transforms.DataTransformFn):
    """Convert Scene 1 dataset inputs to the expected model format."""

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        chest_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": chest_image,
                "left_wrist_0_rgb": np.zeros_like(chest_image),
                "right_wrist_0_rgb": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Scene1Outputs(transforms.DataTransformFn):
    """Convert model outputs back to Scene 1 action format."""

    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
