import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config, ReCAPConfig


def _config(**recap_kwargs) -> Pi0Config:
    return Pi0Config(
        dtype="float32",
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        recap=ReCAPConfig(enabled=True, value_head=True, **recap_kwargs),
    )


def _batch(config: Pi0Config, batch_size: int = 2):
    obs, actions = config.inputs_spec(batch_size=batch_size)
    obs = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), obs)
    adv_mask = np.zeros((batch_size, 8), dtype=bool)
    adv_mask[:, :3] = True
    obs = obs.replace(
        advantage_indicator=jnp.array([True, False][:batch_size]),
        use_advantage=jnp.ones((batch_size,), dtype=jnp.bool_),
        tokenized_advantage_mask=jnp.asarray(adv_mask),
        tokenized_advantage_positive=jnp.ones((batch_size, 8), dtype=jnp.int32),
        tokenized_advantage_negative=jnp.full((batch_size, 8), 2, dtype=jnp.int32),
        value_target=jnp.linspace(-0.9, -0.1, batch_size),
    )
    return obs, jnp.ones(actions.shape, actions.dtype)


def test_predict_value_returns_expectation_over_bins():
    config = _config()
    model = config.create(jax.random.key(0))
    obs, _ = _batch(config)

    values, probs = model.predict_value(obs)
    assert values.shape == (2,)
    assert probs.shape == (2, config.recap.value_bins)
    np.testing.assert_allclose(probs.sum(-1), 1.0, atol=1e-5)
    assert ((values >= -1.0) & (values <= 0.0)).all()


def test_value_features_do_not_depend_on_advantage_indicator():
    config = _config()
    model = config.create(jax.random.key(0))
    obs, _ = _batch(config)

    v_pos, _ = model.predict_value(obs.replace(advantage_indicator=jnp.array([True, True])))
    v_neg, _ = model.predict_value(obs.replace(advantage_indicator=jnp.array([False, False])))
    v_none, _ = model.predict_value(obs.replace(use_advantage=jnp.array([False, False])))
    np.testing.assert_allclose(v_pos, v_neg, atol=1e-6)
    np.testing.assert_allclose(v_pos, v_none, atol=1e-6)


@pytest.mark.parametrize("value_mode", ["joint", "value_only"])
def test_recap_loss_with_value_head(value_mode):
    config = _config(value_mode=value_mode)
    model = config.create(jax.random.key(0))
    obs, actions = _batch(config)

    loss = model.compute_recap_loss(jax.random.key(1), obs, actions, train=True)
    assert loss.shape == (2, config.action_horizon)
    assert jnp.isfinite(loss).all()
    if value_mode == "value_only":
        # Value-only loss is a per-sample cross-entropy broadcast over the action horizon.
        np.testing.assert_allclose(loss, jnp.broadcast_to(loss[:, :1], loss.shape))
        assert (loss > 0).all()


def _grad_norms(model, obs, actions):
    def loss_fn(m):
        return jnp.mean(m.compute_recap_loss(jax.random.key(1), obs, actions, train=True))

    grads = nnx.grad(loss_fn)(model)
    flat, _ = jax.tree_util.tree_flatten_with_path(grads)
    head = sum(float(jnp.sum(jnp.square(v))) for k, v in flat if "value_head" in jax.tree_util.keystr(k))
    llm = sum(float(jnp.sum(jnp.square(v))) for k, v in flat if "PaliGemma" in jax.tree_util.keystr(k))
    return head, llm


def test_value_only_trains_backbone_and_head():
    config = _config(value_mode="value_only")
    model = config.create(jax.random.key(0))
    obs, actions = _batch(config)
    head, llm = _grad_norms(model, obs, actions)
    assert head > 0
    assert llm > 0


def test_joint_stop_gradient_isolates_backbone_from_value_loss():
    config = _config(value_mode="joint", value_stop_gradient=True)
    model = config.create(jax.random.key(0))
    obs, actions = _batch(config)

    with_value = _grad_norms(model, obs, actions)
    without_value = _grad_norms(model, obs.replace(value_target=None), actions)
    assert with_value[0] > 0
    assert without_value[0] == 0
    # The backbone gradient comes from the policy loss only.
    np.testing.assert_allclose(with_value[1], without_value[1], rtol=1e-5)


def test_value_head_requires_target():
    config = _config(value_mode="value_only")
    model = config.create(jax.random.key(0))
    obs, actions = _batch(config)
    with pytest.raises(ValueError, match="value_target"):
        model.compute_recap_loss(jax.random.key(1), obs.replace(value_target=None), actions, train=True)


def test_predict_value_requires_value_head():
    config = Pi0Config(
        pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", recap=ReCAPConfig(enabled=True)
    )
    model = config.create(jax.random.key(0))
    with pytest.raises(ValueError, match="value_head"):
        model.predict_value(config.fake_obs(batch_size=1))


def test_observation_round_trips_value_target():
    config = _config()
    obs, _ = _batch(config)
    restored = _model.Observation.from_dict(obs.to_dict())
    np.testing.assert_allclose(restored.value_target, obs.value_target)
