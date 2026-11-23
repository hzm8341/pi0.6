# ruff: noqa: SLF001
import types

import jax.numpy as jnp
import numpy as np

from openpi.models.pi0 import Pi0


def _fake(bins=5):
    return types.SimpleNamespace(value_head=lambda f: f, value_min=-1.0, value_max=0.0, value_bins=bins)


def test_nan_targets_are_masked_and_reweighted():
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(size=(4, 5)), dtype=jnp.float32)
    targets = jnp.asarray([-0.1, jnp.nan, -0.6, jnp.nan])
    ce = Pi0._value_cross_entropy(_fake(), logits, targets)
    assert np.all(np.isfinite(ce))
    assert ce[1] == 0
    assert ce[3] == 0
    ref = Pi0._value_cross_entropy(_fake(), logits[jnp.array([0, 2])], targets[jnp.array([0, 2])])
    np.testing.assert_allclose(ce.mean(), ref.mean(), rtol=1e-6)


def test_all_valid_unchanged():
    logits = jnp.zeros((3, 5))
    ce = Pi0._value_cross_entropy(_fake(), logits, jnp.asarray([-0.2, -0.5, -1.0]))
    np.testing.assert_allclose(ce, np.log(5.0), rtol=1e-6)
