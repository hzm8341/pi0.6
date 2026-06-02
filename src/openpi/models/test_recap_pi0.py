import jax
import jax.numpy as jnp

from openpi.models.pi0_config import Pi0Config, ReCAPConfig


def test_recap_prefix_adds_advantage_tokens_with_dummy_model():
    config = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        recap=ReCAPConfig(enabled=True),
    )
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)
    obs = obs.replace(
        advantage_indicator=jnp.array([True, False]),
        use_advantage=jnp.array([True, True]),
        tokenized_advantage_positive=jnp.ones((2, 8), dtype=jnp.int32),
        tokenized_advantage_negative=jnp.full((2, 8), 2, dtype=jnp.int32),
        tokenized_advantage_mask=jnp.ones((2, 8), dtype=jnp.bool_),
    )

    base_obs = obs.replace(advantage_indicator=None, use_advantage=None)
    base_tokens, _, _ = model.embed_prefix(base_obs)
    recap_tokens, recap_mask, _ = model.embed_prefix(obs)

    assert recap_tokens.shape[1] == base_tokens.shape[1] + 8
    assert recap_mask[:, -8:].all()


def test_compute_recap_loss_returns_per_chunk_loss():
    config = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        recap=ReCAPConfig(enabled=True),
    )
    model = config.create(jax.random.key(0))
    obs, actions = config.inputs_spec(batch_size=2)
    obs = jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), obs)
    actions = jnp.ones(actions.shape, actions.dtype)
    obs = obs.replace(
        advantage_indicator=jnp.array([True, False]),
        use_advantage=jnp.array([True, True]),
        tokenized_advantage_positive=jnp.ones((2, 8), dtype=jnp.int32),
        tokenized_advantage_negative=jnp.full((2, 8), 2, dtype=jnp.int32),
        tokenized_advantage_mask=jnp.ones((2, 8), dtype=jnp.bool_),
    )

    loss = model.compute_recap_loss(jax.random.key(1), obs, actions, train=True)
    assert loss.shape == (2, config.action_horizon)
