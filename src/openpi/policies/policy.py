from collections.abc import Sequence
from collections import deque
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results


# ============================================================================
# π0.6-MEM: policy with frame-buffer and language-memory management
# ============================================================================

class MEMPolicy(Policy):
    """Extends ``Policy`` with short-term frame-buffer and language-memory.

    At each ``infer`` call it:
    1. Builds a ``image_history`` dict from the rolling frame deque.
    2. Injects ``tokenized_memory`` / ``tokenized_memory_mask`` if a
       ``HighLevelPolicy`` is attached.
    3. Calls the base ``Policy.infer`` with the augmented observation.
    4. Appends the current frame to the deque (AFTER inference).
    5. Optionally triggers a high-level policy update.

    Parameters
    ----------
    model:
        The π0.6-MEM JAX model.
    high_level_policy:
        Optional ``HighLevelPolicy`` instance.  When provided language memory
        is automatically maintained.
    num_video_frames:
        K – total frames (current + K-1 history).
    camera_keys:
        Camera names to buffer.
    **kwargs:
        Forwarded verbatim to the parent ``Policy`` constructor.
    """

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        high_level_policy: Any | None = None,   # HighLevelPolicy | None
        num_video_frames: int = 6,
        camera_keys: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self.K = num_video_frames
        self.camera_keys = camera_keys or [
            "base_0_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
        ]
        self._hl = high_level_policy

        # Rolling frame buffers (maxlen = K-1 history frames)
        self._frame_buf: dict[str, deque] = {
            k: deque(maxlen=max(self.K - 1, 1)) for k in self.camera_keys
        }
        self._state_buf: deque = deque(maxlen=max(self.K - 1, 1))
        self._language_memory: str = ""
        self._episode_step: int = 0

    # ------------------------------------------------------------------
    def reset_episode(self, task_goal: str = "") -> None:
        """Clear all buffers and reset the high-level policy."""
        for buf in self._frame_buf.values():
            buf.clear()
        self._state_buf.clear()
        self._language_memory = ""
        self._episode_step = 0
        if self._hl is not None and task_goal:
            self._hl.reset(task_goal)

    # ------------------------------------------------------------------
    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Step 1 – inject history into the observation dict
        obs_aug = self._build_obs_with_history(obs)

        # Step 2 – inject language memory tokens
        if self._hl is not None and self._language_memory:
            token_ids, token_mask = self._hl.tokenize_memory(self._language_memory)
            obs_aug["tokenized_memory"] = token_ids
            obs_aug["tokenized_memory_mask"] = token_mask

        # Step 3 – base inference
        result = super().infer(obs_aug, noise=noise)

        # Step 4 – update buffers AFTER inference
        self._push_to_buffers(obs)
        self._episode_step += 1

        # Step 5 – high-level policy update (if due)
        if self._hl is not None and self._hl.should_update():
            base_img = obs.get("base_0_rgb")
            if base_img is not None:
                _, new_memory = self._hl.update(
                    observation_image=np.asarray(base_img),
                    subtask_success=True,
                )
                self._language_memory = new_memory

        return result

    # ------------------------------------------------------------------
    def _build_obs_with_history(self, obs: dict) -> dict:
        """Return a copy of *obs* augmented with ``image_history`` / ``state_history``."""
        result = dict(obs)

        if self.K <= 1:
            return result

        image_history: dict[str, np.ndarray] = {}
        for cam in self.camera_keys:
            if cam not in obs:
                continue
            buf = list(self._frame_buf[cam])
            n_needed = self.K - 1
            if len(buf) == 0:
                # No history yet – replicate the current frame as padding
                pad = obs[cam]
                buf = [pad] * n_needed
            elif len(buf) < n_needed:
                # Partial history – pad with the oldest available frame
                pad = buf[0]
                buf = [pad] * (n_needed - len(buf)) + buf
            image_history[cam] = np.stack(buf, axis=0)  # (K-1, H, W, C)

        if image_history:
            result["image_history"] = image_history

        state_buf = list(self._state_buf)
        if state_buf:
            n_needed = self.K - 1
            if len(state_buf) < n_needed:
                state_buf = [state_buf[0]] * (n_needed - len(state_buf)) + state_buf
            result["state_history"] = np.stack(state_buf, axis=0)  # (K-1, S)

        return result

    def _push_to_buffers(self, obs: dict) -> None:
        """Push current observation into the rolling buffers."""
        for cam in self.camera_keys:
            if cam in obs:
                self._frame_buf[cam].append(np.asarray(obs[cam]).copy())
        if "state" in obs:
            self._state_buf.append(np.asarray(obs["state"]).copy())

