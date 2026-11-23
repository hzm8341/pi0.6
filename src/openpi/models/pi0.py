import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


class ValueHead(nnx.Module):
    """Two-layer MLP → value-bin logits. Named sub-modules (not nnx.Sequential) so parameter paths
    survive a checkpoint round trip (Sequential's integer list keys come back as strings)."""

    def __init__(self, in_dim: int, hidden_dim: int, num_bins: int, *, rngs: nnx.Rngs):
        self.hidden = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.out = nnx.Linear(hidden_dim, num_bins, rngs=rngs)

    def __call__(self, x):
        return self.out(nnx.swish(self.hidden(x)))


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])

        # ── MEM: video encoder uses SpaceTimeSeparableBlock when K > 1 ────────
        K = config.video_memory_frames if config.use_video_memory else 1
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=(K == 1),          # scan only for single-frame (π0.5 compat)
                dtype_mm=config.dtype,
                num_timesteps=K,
                temporal_attn_every=config.mem.temporal_attn_every_n_layers,
                drop_history_after_layer=config.mem.drop_history_tokens_after_layer,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # ── MEM flags (stored as plain Python attrs, not nnx params) ──────────
        self.use_video_memory = config.use_video_memory
        self.use_language_memory = config.use_language_memory
        self.use_state_history = config.use_state_history
        self.num_video_frames = K
        self.recap_enabled = config.recap.enabled
        self.recap_alpha = config.recap.alpha
        self.recap_advantage_dropout_prob = config.recap.advantage_dropout_prob
        if config.recap.loss_mode not in ("single", "dual"):
            raise ValueError(f"Unknown RECAP loss_mode: {config.recap.loss_mode!r}")
        self.recap_loss_mode = config.recap.loss_mode
        self.recap_cfg_beta = float(config.recap.cfg_beta)

        # RECAP value head on the shared backbone (plan feature F1).
        recap = config.recap
        self.value_head_enabled = recap.enabled and recap.value_head
        if self.value_head_enabled:
            if recap.value_mode not in ("joint", "value_only"):
                raise ValueError(f"Unknown RECAP value_mode: {recap.value_mode!r}")
            self.value_mode = recap.value_mode
            self.value_loss_weight = recap.value_loss_weight
            self.value_stop_gradient = recap.value_stop_gradient
            self.value_bins = recap.value_bins
            self.value_min = recap.value_min
            self.value_max = recap.value_max
            self.value_head = ValueHead(paligemma_config.width, recap.value_hidden_dim, recap.value_bins, rngs=rngs)

        # Linear projection for proprioceptive history (continuous embedding).
        if config.use_video_memory and config.use_state_history:
            self.state_history_proj = nnx.Linear(
                config.action_dim, paligemma_config.width, rngs=rngs
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Encode the prefix tokens.

        Token order (MEM-aware):
          [language-memory tokens]  [image tokens × N_cams]
          [language-prompt tokens]  [state-history tokens]  [RECAP advantage tokens]

        When MEM and RECAP are disabled this is identical to π0.5.
        """
        tokens, input_mask, ar_mask, _ = self._embed_prefix_with_advantage(obs)
        return tokens, input_mask, ar_mask

    def _embed_prefix_with_advantage(self, obs: _model.Observation):
        """Like `embed_prefix`, but also returns a (b, s) mask marking the RECAP advantage tokens.

        The advantage tokens form their own causal block at the end of the prefix: they attend to
        everything before them, but image / prompt tokens cannot attend to them. This keeps the
        observation representation independent of the indicator (so a value head can read it and
        conditional / unconditional CFG branches can share the prefix KV cache), and matches the
        paper, where "Advantage: ..." follows the prompt in a causal sequence.
        """
        input_mask = []
        ar_mask = []
        tokens = []
        advantage_positions = []

        # ── (A) Long-term language memory ────────────────────────────────────
        if self.use_language_memory and obs.tokenized_memory is not None:
            memory_tokens = self.PaliGemma.llm(obs.tokenized_memory, method="embed")
            tokens.append(memory_tokens)
            if obs.tokenized_memory_mask is not None:
                input_mask.append(obs.tokenized_memory_mask)
            else:
                input_mask.append(jnp.ones(memory_tokens.shape[:2], dtype=jnp.bool_))
            ar_mask += [False] * memory_tokens.shape[1]

        # ── (B) Image tokens (current frame or video-fused) ──────────────────
        for name in obs.images:
            if self.use_video_memory and obs.image_history is not None and name in obs.image_history:
                image_tokens = self._encode_video_frames(obs.image_history[name], obs.images[name])
            else:
                image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask += [False] * image_tokens.shape[1]

        # ── (C) Language prompt ───────────────────────────────────────────────
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        # ── (D) Proprioceptive history (continuous embedding) ────────────────
        if self.use_video_memory and self.use_state_history and obs.state_history is not None:
            # state_history: (B, K-1, S) → (B, K-1, D)
            state_hist_tokens = self.state_history_proj(obs.state_history)
            tokens.append(state_hist_tokens)
            input_mask.append(jnp.ones(state_hist_tokens.shape[:2], dtype=jnp.bool_))
            ar_mask += [False] * state_hist_tokens.shape[1]

        # ── (E) RECAP advantage condition tokens ─────────────────────────────
        if (
            self.recap_enabled
            and obs.advantage_indicator is not None
            and obs.use_advantage is not None
            and obs.tokenized_advantage_positive is not None
            and obs.tokenized_advantage_negative is not None
            and obs.tokenized_advantage_mask is not None
        ):
            adv_ids = jnp.where(
                obs.advantage_indicator[:, None],
                obs.tokenized_advantage_positive,
                obs.tokenized_advantage_negative,
            )
            adv_tokens = self.PaliGemma.llm(adv_ids, method="embed")
            adv_mask = jnp.logical_and(obs.tokenized_advantage_mask, obs.use_advantage[:, None])
            prefix_len = sum(t.shape[1] for t in tokens)
            tokens.append(adv_tokens)
            input_mask.append(adv_mask)
            ar_mask += [True] + [False] * (adv_tokens.shape[1] - 1)
            advantage_positions = [
                jnp.zeros((adv_tokens.shape[0], prefix_len), dtype=jnp.bool_),
                obs.tokenized_advantage_mask,
            ]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        if advantage_positions:
            advantage_mask = jnp.concatenate(advantage_positions, axis=1)
        else:
            advantage_mask = jnp.zeros(input_mask.shape, dtype=jnp.bool_)
        return tokens, input_mask, ar_mask, advantage_mask

    def _encode_video_frames(
        self,
        image_history: at.Float[at.Array, "b k h w c"],
        current_image: at.Float[at.Array, "b h w c"],
    ) -> at.Float[at.Array, "b n d"]:
        """Encode K frames (K-1 history + 1 current) through the video ViT.

        The video encoder's SpaceTimeSeparableBlock fuses temporal information
        into the current-frame tokens via causal attention.  The upper layers
        of the ViT drop the history tokens, so the output shape is identical
        to single-frame encoding: (B, N, D).

        Args:
            image_history: (B, K-1, H, W, C)
            current_image:  (B, H, W, C)
        Returns:
            tokens: (B, N, D)
        """
        K = self.num_video_frames
        if K <= 1:
            tokens, _ = self.PaliGemma.img(current_image, train=False)
            return tokens

        B = current_image.shape[0]
        # Concatenate along the time axis: (B, K, H, W, C)
        curr = current_image[:, None, :, :, :]           # (B, 1, H, W, C)
        frames = jnp.concatenate([image_history, curr], axis=1)  # (B, K, H, W, C)

        H, W, C = frames.shape[-3:]
        # Flatten to (B*K, H, W, C) for the ViT
        frames_flat = frames.reshape(B * K, H, W, C)
        tokens_flat, _ = self.PaliGemma.img(frames_flat, train=False)

        # The video encoder drops history tokens in the upper layers, so the
        # output is already (B, N, D).  Guard against encoders that haven't
        # implemented drop_history yet (fall back to slicing last frame).
        N, D = tokens_flat.shape[-2], tokens_flat.shape[-1]
        if tokens_flat.shape[0] == B * K:
            # drop_history not applied inside ViT – manually take current frame
            tokens = tokens_flat.reshape(B, K, N, D)[:, -1, :, :]
        else:
            tokens = tokens_flat.reshape(B, N, D)
        return tokens

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        return self._flow_matching_loss(rng, observation, actions, train=train)

    def _flow_matching_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self._flow_matching_loss_and_value_features(rng, observation, actions, train=train)
        return loss

    def _flow_matching_loss_and_value_features(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ):
        """Flow-matching loss plus pooled observation features (b, emb) from the same forward pass."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask, advantage_mask = self._embed_prefix_with_advantage(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        return loss, self._pool_observation_tokens(prefix_out, prefix_mask, advantage_mask)

    # ── RECAP value head ─────────────────────────────────────────────────────
    @staticmethod
    def _pool_observation_tokens(prefix_out, prefix_mask, advantage_mask):
        """Masked mean over valid prefix tokens, excluding the advantage tokens.

        Advantage tokens form a trailing causal block, so the remaining prefix outputs do not depend
        on the indicator: the value estimate is V(o, l), not conditioned on the advantage label.
        """
        mask = jnp.logical_and(prefix_mask, jnp.logical_not(advantage_mask)).astype(jnp.float32)
        summed = jnp.einsum("bsd,bs->bd", prefix_out.astype(jnp.float32), mask)
        return summed / jnp.maximum(mask.sum(axis=1, keepdims=True), 1.0)

    def _observation_features(self, observation: _model.Observation):
        """Prefix-only forward pass → pooled observation features (b, emb). Expects a preprocessed observation."""
        prefix_tokens, prefix_mask, prefix_ar_mask, advantage_mask = self._embed_prefix_with_advantage(observation)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
        return self._pool_observation_tokens(prefix_out, prefix_mask, advantage_mask)

    def _value_bin_centers(self):
        return jnp.linspace(self.value_min, self.value_max, self.value_bins)

    def _value_cross_entropy(self, features, value_target):
        """Cross-entropy against the nearest value bin (paper: discretized empirical returns, 201 bins).

        NaN targets are masked out (held-out frames for value evaluation); the remaining frames are reweighted so
        the batch mean equals the mean over valid frames.
        """
        logits = self.value_head(features).astype(jnp.float32)
        target = value_target.astype(jnp.float32)
        valid = jnp.isfinite(target)
        target = jnp.where(valid, target, self.value_min)
        clipped = jnp.clip(target, self.value_min, self.value_max)
        scaled = (clipped - self.value_min) / (self.value_max - self.value_min)
        target_bins = jnp.round(scaled * (self.value_bins - 1)).astype(jnp.int32)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce = -jnp.take_along_axis(log_probs, target_bins[:, None], axis=-1)[:, 0]
        weight = valid.astype(jnp.float32) * (valid.shape[0] / jnp.maximum(valid.sum(), 1))
        return ce * weight

    def predict_value(self, observation: _model.Observation):
        """Returns (expected value (b,), bin probabilities (b, bins)) for raw (unpreprocessed) observations."""
        if not self.value_head_enabled:
            raise ValueError("predict_value requires ReCAPConfig(enabled=True, value_head=True).")
        observation = _model.preprocess_observation(None, observation, train=False)
        logits = self.value_head(self._observation_features(observation)).astype(jnp.float32)
        probs = jax.nn.softmax(logits, axis=-1)
        return probs @ self._value_bin_centers(), probs

    def compute_value_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, *, train: bool = False
    ) -> at.Float[at.Array, " b"]:
        """Value-only objective: prefix forward pass + cross-entropy (gradients reach the backbone)."""
        if observation.value_target is None:
            raise ValueError("Value training requires `value_target` in the batch.")
        observation = _model.preprocess_observation(rng, observation, train=train)
        return self._value_cross_entropy(self._observation_features(observation), observation.value_target)

    def compute_recap_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        dropout_rng, uncond_rng, cond_rng = jax.random.split(rng, 3)
        batch_size = actions.shape[0]
        keep = jax.random.bernoulli(
            dropout_rng,
            p=1.0 - self.recap_advantage_dropout_prob,
            shape=(batch_size,),
        )
        if observation.use_advantage is not None:
            keep = jnp.logical_and(keep, observation.use_advantage)
        cond_obs = observation.replace(use_advantage=keep)

        if self.value_head_enabled and self.value_mode == "value_only":
            value_loss = self.compute_value_loss(cond_rng, observation, train=train)
            return jnp.broadcast_to(value_loss[:, None], actions.shape[:-1])

        if self.recap_loss_mode == "single":
            loss, features = self._flow_matching_loss_and_value_features(cond_rng, cond_obs, actions, train=train)
            return loss + self._joint_value_loss(features, observation)[:, None]

        uncond_obs = observation.replace(use_advantage=jnp.zeros((batch_size,), dtype=jnp.bool_))
        loss_uncond = self._flow_matching_loss(uncond_rng, uncond_obs, actions, train=train)
        loss_cond, features = self._flow_matching_loss_and_value_features(cond_rng, cond_obs, actions, train=train)
        return loss_uncond + self.recap_alpha * loss_cond + self._joint_value_loss(features, observation)[:, None]

    def _joint_value_loss(self, features, observation: _model.Observation):
        """Weighted value cross-entropy for joint training, or zeros when there is no value head / target."""
        batch_size = features.shape[0]
        if not self.value_head_enabled or observation.value_target is None:
            return jnp.zeros((batch_size,), dtype=jnp.float32)
        if self.value_stop_gradient:
            features = jax.lax.stop_gradient(features)
        return self.value_loss_weight * self._value_cross_entropy(features, observation.value_target)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        cfg_beta: float | at.Float[at.Array, ""] | None = None,
    ) -> _model.Actions:
        """Sample actions with flow matching.

        For RECAP models, `cfg_beta` (default: `config.recap.cfg_beta`) controls classifier-free
        guidance over the advantage indicator. 1.0 samples from the conditioned policy only;
        values > 1 combine conditional and unconditional velocities. Both branches share the
        prefix KV cache: they differ only in whether the suffix may attend to the advantage tokens.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        if cfg_beta is None:
            cfg_beta = self.recap_cfg_beta
            use_cfg = self.recap_enabled and cfg_beta != 1.0
        else:
            use_cfg = self.recap_enabled
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask, advantage_mask = self._embed_prefix_with_advantage(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        # The unconditional branch hides the advantage tokens. Because those tokens sit in their own
        # causal block at the end of the prefix, the KV entries of every other token are unchanged.
        uncond_prefix_mask = jnp.logical_and(prefix_mask, jnp.logical_not(advantage_mask))

        def velocity(x_t, time, prefix_mask):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        def step(carry):
            x_t, time = carry
            v_t = velocity(x_t, time, prefix_mask)
            if use_cfg:
                # Combine in float32: in bfloat16, v_u + 1 * (v_c - v_u) visibly differs from v_c.
                v_cond = v_t.astype(jnp.float32)
                v_uncond = velocity(x_t, time, uncond_prefix_mask).astype(jnp.float32)
                v_t = (v_uncond + cfg_beta * (v_cond - v_uncond)).astype(x_t.dtype)
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
