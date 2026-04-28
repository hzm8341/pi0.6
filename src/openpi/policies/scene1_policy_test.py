import numpy as np

from openpi.models import model as _model
from openpi.policies import scene1_policy


def test_scene1_inputs_maps_chest_and_wrist_images():
    data = scene1_policy.make_scene1_example()
    inputs = scene1_policy.Scene1Inputs(model_type=_model.ModelType.PI05)(data)

    assert inputs["state"].shape == (8,)
    assert list(inputs["image"].keys()) == ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    np.testing.assert_array_equal(inputs["image"]["base_0_rgb"], data["observation/image"])
    np.testing.assert_array_equal(inputs["image"]["right_wrist_0_rgb"], data["observation/wrist_image"])
    np.testing.assert_array_equal(inputs["image"]["left_wrist_0_rgb"], np.zeros_like(data["observation/image"]))
    assert bool(inputs["image_mask"]["base_0_rgb"])
    assert not bool(inputs["image_mask"]["left_wrist_0_rgb"])
    assert bool(inputs["image_mask"]["right_wrist_0_rgb"])
    assert inputs["prompt"] == data["prompt"]
    assert inputs["actions"].shape == (8,)


def test_scene1_outputs_trims_to_dataset_action_dim():
    data = {"actions": np.arange(32, dtype=np.float32).reshape(4, 8)}
    outputs = scene1_policy.Scene1Outputs()(data)

    np.testing.assert_array_equal(outputs["actions"], data["actions"])
