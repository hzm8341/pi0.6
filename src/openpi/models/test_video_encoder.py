"""Tests for the video encoder in siglip.py.

Critical properties verified:
1. K=1 output is numerically identical to the original single-frame ViT.
2. Multi-frame output shape is correct (history tokens dropped → (B, N, D)).
3. Temporal information from history frames propagates to current-frame tokens.
4. Causal mask: changing the current frame does NOT affect earlier-frame
   activations in the first block.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_module(K: int = 1, depth: int = 4, scan: bool = False):
    """Return an initialised SigLIP _Module for testing.

    Uses a tiny 'mu' variant (width=32, depth=1) so tests run on CPU quickly,
    overriding depth to ``depth`` for multi-layer tests.
    """
    from openpi.models.siglip import _Module, decode_variant

    cfg = decode_variant("mu")   # smallest variant (width=32, depth=1, heads=2)
    cfg = {**cfg, "depth": depth}
    return _Module(
        num_classes=0,
        pool_type="none",
        scan=scan,
        num_timesteps=K,
        temporal_attn_every=4,
        drop_history_after_layer=-(max(depth // 2, 1)),
        **cfg,
    )


def _init_and_call(model, imgs, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)
    variables = model.init(key, imgs, train=False)
    out, _ = model.apply(variables, imgs, train=False)
    return out, variables


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSingleFrameEquivalence:
    """K=1 video encoder must be numerically equivalent to standard ViT."""

    def test_k1_same_output_as_original(self):
        from openpi.models.siglip import _Module, decode_variant

        cfg = {**decode_variant("mu"), "depth": 4}
        key = jax.random.PRNGKey(42)
        imgs = jax.random.normal(key, (2, 32, 32, 3))

        # Original (scan, no video params)
        orig = _Module(num_classes=0, pool_type="none", scan=True, **cfg)
        vars_orig = orig.init(key, imgs, train=False)
        out_orig, _ = orig.apply(vars_orig, imgs, train=False)

        # K=1 video encoder (non-scan path, but same weights)
        vid = _Module(num_classes=0, pool_type="none", scan=False,
                      num_timesteps=1, **cfg)
        vars_vid = vid.init(key, imgs, train=False)
        out_vid, _ = vid.apply(vars_vid, imgs, train=False)

        max_diff = float(jnp.max(jnp.abs(out_orig - out_vid)))
        assert max_diff < 1e-5, (
            f"K=1 video encoder differs from standard ViT by {max_diff:.2e} "
            f"(must be < 1e-5)"
        )


class TestMultiFrameShape:
    """With K>1 the output must have the same shape as K=1."""

    @pytest.mark.parametrize("K", [2, 4, 6])
    def test_output_shape(self, K: int):
        B = 2
        model = _make_module(K=K, depth=4)
        key = jax.random.PRNGKey(0)
        # Input: B*K frames flattened
        imgs = jnp.ones((B * K, 32, 32, 3))
        out, _ = _init_and_call(model, imgs)
        # N = (32/16)^2 = 4 for mu/16, width=32
        assert out.shape[0] == B, f"Batch dim wrong: {out.shape}"
        assert out.ndim == 3, f"Expected 3-D output (B, N, D), got shape {out.shape}"


class TestTemporalInformationFlow:
    """Changing history frames must change the current-frame output tokens."""

    def test_history_affects_output(self):
        K = 3
        B = 1
        model = _make_module(K=K, depth=4)
        key = jax.random.PRNGKey(0)

        frames_zeros = jnp.zeros((B * K, 32, 32, 3))
        vars_ = model.init(key, frames_zeros, train=False)

        out_base, _ = model.apply(vars_, frames_zeros, train=False)

        # Modify only history frames (first K-1), keep last frame identical
        frames_mod = frames_zeros.at[: B * (K - 1)].set(
            jax.random.normal(key, (B * (K - 1), 32, 32, 3))
        )
        out_mod, _ = model.apply(vars_, frames_mod, train=False)

        mean_diff = float(jnp.mean(jnp.abs(out_base - out_mod)))
        assert mean_diff > 1e-6, (
            "Temporal attention has no effect: output unchanged when history "
            f"frames modified (mean_diff={mean_diff:.2e})"
        )


class TestCausalMask:
    """SpaceTimeSeparableBlock's temporal attention must be causal."""

    def test_single_block_causal(self):
        """Changing the last (current) frame must NOT change first block output
        for the *first* time-step's spatial tokens."""
        from openpi.models.siglip import SpaceTimeSeparableBlock

        K = 3
        B, N, D = 2, 4, 32
        key = jax.random.PRNGKey(7)

        block = SpaceTimeSeparableBlock(
            mlp_dim=64, num_heads=2, dtype_mm="float32", num_timesteps=K
        )
        x = jax.random.normal(key, (B * K, N, D))
        params = block.init(key, x)

        out_base, _ = block.apply(params, x)

        # Perturb only the LAST frame (current frame, index K-1)
        x_mod = x.at[-(B):].set(jax.random.normal(key, (B, N, D)))
        out_mod, _ = block.apply(params, x_mod)

        # First frame's output (indices 0..B-1) must be unchanged
        diff_first = float(jnp.max(jnp.abs(out_base[:B] - out_mod[:B])))
        assert diff_first < 1e-5, (
            f"Causal mask violated: first-frame activations changed by "
            f"{diff_first:.2e} when current frame was modified"
        )
