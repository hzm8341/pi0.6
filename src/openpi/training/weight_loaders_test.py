import flax.traverse_util as traverse_util
import jax
import numpy as np
import pytest

from openpi.models import siglip
from openpi.training import weight_loaders

_DEPTH = 4


def _vision_params(*, num_timesteps: int, depth: int = _DEPTH, abstract: bool) -> dict:
    module = siglip._Module(  # noqa: SLF001
        num_classes=0,
        pool_type="none",
        scan=num_timesteps == 1,
        num_timesteps=num_timesteps,
        temporal_attn_every=2,
        drop_history_after_layer=-1,
        **{**siglip.decode_variant("mu"), "depth": depth},
    )
    images = np.zeros((num_timesteps, 32, 32, 3), np.float32)
    init = lambda: module.init(jax.random.key(0), images, train=False)["params"]  # noqa: E731
    params = jax.eval_shape(init) if abstract else jax.tree.map(np.asarray, init())
    return {"PaliGemma": {"img": params}}


def test_stacked_checkpoint_loads_into_per_layer_video_encoder():
    checkpoint = _vision_params(num_timesteps=1, abstract=False)
    reference = _vision_params(num_timesteps=4, abstract=True)

    merged = traverse_util.flatten_dict(
        weight_loaders._merge_params(checkpoint, reference, missing_regex=".*lora.*"),
        sep="/",  # noqa: SLF001
    )

    stacked = traverse_util.flatten_dict(checkpoint, sep="/")
    prefix = "PaliGemma/img/Transformer/encoderblock"
    per_layer_keys = [k for k in traverse_util.flatten_dict(reference, sep="/") if k.startswith(prefix + "_")]
    temporal = [k for k in per_layer_keys if "/LayerNorm_2/" in k or "/MultiHeadDotProductAttention_1/" in k]
    pretrained = [k for k in per_layer_keys if k not in temporal]

    assert temporal, "reference has temporal-attention params"
    assert not any(k in merged for k in temporal), "temporal params stay for random init"
    for key in pretrained:
        layer, rest = key[len(prefix) + 1 :].split("/", 1)
        np.testing.assert_array_equal(merged[key], stacked[f"{prefix}/{rest}"][int(layer)], err_msg=key)


def test_stacked_checkpoint_with_wrong_depth_is_rejected():
    checkpoint = _vision_params(num_timesteps=1, depth=_DEPTH - 1, abstract=False)
    reference = _vision_params(num_timesteps=4, abstract=True)

    with pytest.raises(ValueError, match="encoderblock"):
        weight_loaders._merge_params(checkpoint, reference, missing_regex=".*lora.*")  # noqa: SLF001
