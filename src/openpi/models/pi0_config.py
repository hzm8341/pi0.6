import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class MEMConfig:
    """Multi-Scale Embodied Memory configuration.

    All flags default to False / 1, so an unmodified ``Pi0Config`` is
    completely equivalent to the original π0.5.
    """

    # ── Short-term visual memory (video encoder) ─────────────────────────────
    # Enable the video encoder.  When False every MEM branch is skipped.
    use_video_memory: bool = False
    # Total number of frames K fed to the video encoder (including current frame).
    # Pre-training typically uses 6; fine-tuning can extend up to 18.
    video_memory_frames: int = 6
    # Wall-clock stride between history frames (seconds), used by the data loader.
    video_frame_stride_sec: float = 1.0
    # Insert SpaceTimeSeparableBlock every N ViT layers.
    temporal_attn_every_n_layers: int = 4
    # Drop history tokens after this ViT layer (negative = from end).
    drop_history_tokens_after_layer: int = -4

    # ── Long-term language memory ─────────────────────────────────────────────
    use_language_memory: bool = False
    # Maximum number of tokens allocated to the language memory.
    max_memory_tokens: int = 256
    # Weight of the memory-prediction cross-entropy loss.
    memory_loss_weight: float = 0.1

    # ── Proprioceptive history embedding ─────────────────────────────────────
    # When True, K-1 past states are embedded via a linear projection (not as
    # text tokens), keeping token budget small.
    use_state_history: bool = False
    # Number of history state frames (typically == video_memory_frames).
    state_history_frames: int = 6


@dataclasses.dataclass(frozen=True)
class ReCAPConfig:
    enabled: bool = False
    # Weight of the conditioned term. Only used when loss_mode == "dual".
    alpha: float = 1.0
    # Probability of dropping the advantage indicator during training so the
    # model also learns the unconditional distribution (needed for CFG).
    advantage_dropout_prob: float = 0.1
    # "single": one forward pass per step, the indicator is dropped per-sample
    #           with `advantage_dropout_prob` (RLinf / pistar style, ~2x cheaper).
    # "dual":   paper Eq. 3 written literally: unconditional loss + alpha * conditional loss.
    loss_mode: str = "single"
    # Inference guidance. 1.0 = sample from the positive-conditioned policy only
    # (paper default). >1.0 enables classifier-free guidance:
    #   v = v_uncond + cfg_beta * (v_cond - v_uncond).
    cfg_beta: float = 1.0

    # ── Value head (shared π0.5 backbone, plan feature F1) ──────────────────
    # Adds a distributional value head on the pooled observation tokens of the prefix
    # (image + prompt/state; the causal advantage tokens are excluded).
    value_head: bool = False
    # "joint": policy flow-matching loss + value_loss_weight * value cross-entropy.
    # "value_only": prefix forward + value cross-entropy only (train a value checkpoint for labeling).
    value_mode: str = "joint"
    value_loss_weight: float = 0.1
    # Joint mode only: stop value gradients from flowing into the shared backbone, so the value
    # head cannot disturb the policy representation.
    value_stop_gradient: bool = True
    value_hidden_dim: int = 512
    positive_fraction: float = 0.4
    value_bins: int = 201
    value_min: float = -1.0
    value_max: float = 0.0
    n_step_lookahead: int = 50


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # π0.6-MEM configuration (defaults → π0.5 compatible)
    mem: MEMConfig = dataclasses.field(default_factory=MEMConfig)
    recap: ReCAPConfig = dataclasses.field(default_factory=ReCAPConfig)

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    # ── Convenience pass-throughs from MEMConfig ────────────────────────────
    @property
    def use_video_memory(self) -> bool:
        return self.mem.use_video_memory

    @property
    def use_language_memory(self) -> bool:
        return self.mem.use_language_memory

    @property
    def video_memory_frames(self) -> int:
        return self.mem.video_memory_frames

    @property
    def use_state_history(self) -> bool:
        return self.mem.use_state_history

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
            if self.recap.enabled:
                adv_len = 8
                observation_spec = observation_spec.replace(
                    advantage_indicator=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                    use_advantage=jax.ShapeDtypeStruct([batch_size], jnp.bool_),
                    tokenized_advantage_positive=jax.ShapeDtypeStruct([batch_size, adv_len], jnp.int32),
                    tokenized_advantage_negative=jax.ShapeDtypeStruct([batch_size, adv_len], jnp.int32),
                    tokenized_advantage_mask=jax.ShapeDtypeStruct([batch_size, adv_len], jnp.bool_),
                )
                if self.recap.value_head:
                    observation_spec = observation_spec.replace(
                        value_target=jax.ShapeDtypeStruct([batch_size], jnp.float32)
                    )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
