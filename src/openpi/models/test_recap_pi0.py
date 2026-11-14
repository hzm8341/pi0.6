import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_config import Pi0Config, ReCAPConfig

ADV_LEN = 8


@pytest.fixture(autouse=True)
def _highest_matmul_precision():
    # On GPU, float32 matmuls default to TF32; separately compiled programs (conditional-only vs. CFG)
    # then differ by ~1e-3, which breaks the exact-equivalence checks below.
    with jax.default_matmul_precision("highest"):
        yield


def _config(**recap_kwargs) -> Pi0Config:
    # float32: in bfloat16 on GPU, separately compiled programs differ by ~1e-2, which is as large as the
    # conditioning effect of the perturbed dummy model and makes equality checks meaningless.
    return Pi0Config(
        dtype="float32",
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        recap=ReCAPConfig(enabled=True, **recap_kwargs),
    )


def _with_advantage(obs, indicator=(True, False), use=(True, True)):
    batch = len(indicator)
    adv_mask = np.zeros((batch, ADV_LEN), dtype=bool)
    adv_mask[:, :3] = True
    return obs.replace(
        advantage_indicator=jnp.array(indicator),
        use_advantage=jnp.array(use),
        tokenized_advantage_positive=jnp.ones((batch, ADV_LEN), dtype=jnp.int32),
        tokenized_advantage_negative=jnp.full((batch, ADV_LEN), 2, dtype=jnp.int32),
        tokenized_advantage_mask=jnp.asarray(adv_mask),
    )


def test_recap_prefix_adds_advantage_tokens_with_dummy_model():
    config = _config()
    model = config.create(jax.random.key(0))
    obs = _with_advantage(config.fake_obs(batch_size=2))

    base_obs = obs.replace(advantage_indicator=None, use_advantage=None)
    base_tokens, _, _ = model.embed_prefix(base_obs)
    recap_tokens, recap_mask, _ = model.embed_prefix(obs)

    assert recap_tokens.shape[1] == base_tokens.shape[1] + ADV_LEN
    assert recap_mask[:, -ADV_LEN:][:, :3].all()
    assert not recap_mask[:, -ADV_LEN:][:, 3:].any()


def test_advantage_tokens_form_causal_block():
    config = _config()
    model = config.create(jax.random.key(0))
    obs = _with_advantage(config.fake_obs(batch_size=2))

    _, input_mask, ar_mask, adv_positions = model._embed_prefix_with_advantage(obs)
    assert ar_mask[-ADV_LEN] and not ar_mask[-ADV_LEN + 1 :].any()
    assert not ar_mask[: -ADV_LEN].any()
    assert adv_positions[:, -ADV_LEN:][:, :3].all()
    assert not adv_positions[:, : -ADV_LEN].any()

    attn = make_attn_mask(input_mask, ar_mask)
    # Observation tokens never attend to advantage tokens; advantage tokens attend to observation tokens.
    assert not attn[:, : -ADV_LEN, -ADV_LEN:].any()
    assert attn[:, -ADV_LEN, 0].all()


@pytest.mark.parametrize("loss_mode", ["single", "dual"])
def test_compute_recap_loss_returns_per_chunk_loss(loss_mode):
    config = _config(loss_mode=loss_mode)
    model = config.create(jax.random.key(0))
    obs, actions = config.inputs_spec(batch_size=2)
    obs = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), obs)
    actions = jnp.ones(actions.shape, actions.dtype)
    obs = _with_advantage(obs)

    loss = model.compute_recap_loss(jax.random.key(1), obs, actions, train=True)
    assert loss.shape == (2, config.action_horizon)
    assert jnp.isfinite(loss).all()


def test_invalid_loss_mode_raises():
    with pytest.raises(ValueError, match="loss_mode"):
        _config(loss_mode="bogus").create(jax.random.key(0))


def _perturbed(model, scale: float = 0.05):
    """π0.5 zero-initialises its adaRMS gates, so at init the action expert ignores the prefix.
    Perturb all float parameters so conditioning has a measurable effect."""
    graphdef, state = nnx.split(model)
    leaves, treedef = jax.tree.flatten(state)
    keys = jax.random.split(jax.random.key(7), len(leaves))
    leaves = [
        leaf + scale * jax.random.normal(key, leaf.shape, leaf.dtype) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf
        for leaf, key in zip(leaves, keys, strict=True)
    ]
    return nnx.merge(graphdef, jax.tree.unflatten(treedef, leaves))


def _sample(model, obs, noise, **kwargs):
    return model.sample_actions(jax.random.key(0), obs, num_steps=3, noise=noise, **kwargs)


def test_cfg_beta_interpolates_between_unconditional_and_conditional():
    config = _config()
    model = _perturbed(config.create(jax.random.key(0)))
    obs = _with_advantage(config.fake_obs(batch_size=2), indicator=(True, True))
    noise = jax.random.normal(jax.random.key(3), (2, config.action_horizon, config.action_dim))

    cond = _sample(model, obs, noise)
    cond_beta1 = _sample(model, obs, noise, cfg_beta=1.0)
    uncond = _sample(model, obs.replace(use_advantage=jnp.array([False, False])), noise)
    beta0 = _sample(model, obs, noise, cfg_beta=0.0)
    beta2 = _sample(model, obs, noise, cfg_beta=2.0)

    np.testing.assert_allclose(cond_beta1, cond, atol=1e-5)
    np.testing.assert_allclose(beta0, uncond, atol=1e-5)
    assert jnp.isfinite(beta2).all()
    # Conditioning must actually change the output, otherwise the tests above are vacuous.
    assert not np.allclose(cond, uncond, atol=1e-5)


def test_config_cfg_beta_is_used_by_default():
    config = _config(cfg_beta=0.0)
    model = _perturbed(config.create(jax.random.key(0)))
    obs = _with_advantage(config.fake_obs(batch_size=2), indicator=(True, True))
    noise = jax.random.normal(jax.random.key(3), (2, config.action_horizon, config.action_dim))

    uncond = _sample(model, obs.replace(use_advantage=jnp.array([False, False])), noise)
    np.testing.assert_allclose(_sample(model, obs, noise), uncond, atol=1e-5)
