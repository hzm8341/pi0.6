# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A refactored and simplified ViT adoptation for Pi, taken from big_vision."""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.sharding as sharding


def posemb_sincos_2d(h, w, width, temperature=10_000.0, dtype=jnp.float32):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1.0 / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
    if typ == "learn":
        return self.param(
            name,
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            (1, np.prod(seqshape), width),
            dtype,
        )
    if typ == "sincos2d":
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """Applies Transformer MlpBlock module."""
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }

        _, _, d = x.shape  # n,l,d
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)


class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)
        return x, out


class SpaceTimeSeparableBlock(nn.Module):
    """Transformer block with factorised spatial + causal-temporal attention (MEM).

    Replaces every ``temporal_attn_every``-th spatial block in the ViT encoder.
    After the standard spatial MHSA + MLP, a *separate* causal temporal attention
    pass lets each patch attend to its own representation across past timesteps.

    Key properties
    --------------
    * No new learnable parameters – reuses the same QKV projections as the
      spatial attention.
    * When K = 1 the sinusoidal temporal position embedding is identically zero
      (boundary condition e(t=0)=0 after the shift), so the temporal attention
      degenerates to self-attention on a single token and adds zero information –
      making this block numerically equivalent to Encoder1DBlock for K=1.
    * Computational complexity: O(K·n² + n·K²) vs O(n²·K²) for joint attention.
    """

    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"
    num_timesteps: int = 1  # K (total frames, including current)

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """Args:
            x: (B*K, N, D) – batch×frames flattened into leading dim.
        Returns:
            x: (B*K, N, D)
        """
        K = self.num_timesteps
        BK, N, D = x.shape
        B = BK // K

        # ── Step 1: Standard spatial attention (identical to Encoder1DBlock) ──
        out = {}
        x = sharding.activation_sharding_constraint(x)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)

        # ── Step 2: Causal temporal attention (MEM-specific) ──
        if K > 1:
            # (B*K, N, D) → (B, K, N, D) → (B*N, K, D)
            x_t = x.reshape(B, K, N, D)
            x_t = x_t.transpose(0, 2, 1, 3).reshape(B * N, K, D)

            # Sinusoidal temporal position encoding, e(t=0)=0 after shift
            time_pos = self._sinusoidal_time_emb(K, D)  # (K, D)
            x_t = x_t + time_pos[None, :, :]

            # Lower-triangular causal mask: timestep t attends to t' ≤ t
            causal_mask = jnp.tril(jnp.ones((K, K), dtype=jnp.bool_))[None, :, :]  # (1, K, K)

            residual_t = x_t
            y_t = nn.LayerNorm(dtype=self.dtype_mm)(x_t)
            y_t = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                kernel_init=nn.initializers.xavier_uniform(),
                deterministic=deterministic,
                dtype=self.dtype_mm,
            )(y_t, y_t, mask=causal_mask)
            y_t = nn.Dropout(rate=self.dropout)(y_t, deterministic)
            x_t = residual_t + y_t  # (B*N, K, D)

            # (B*N, K, D) → (B, N, K, D) → (B, K, N, D) → (B*K, N, D)
            x = x_t.reshape(B, N, K, D).transpose(0, 2, 1, 3).reshape(B * K, N, D)

        return x, out

    def _sinusoidal_time_emb(self, K: int, D: int):
        """Sinusoidal position encoding with e(t=0)=0 (shift by t=0 value).

        This guarantees that with K=1 the temporal attention adds nothing,
        keeping numerical equivalence with the single-frame ViT.
        """
        positions = jnp.arange(K, dtype=jnp.float32)
        half_d = D // 2
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half_d) / (half_d - 1))
        args = positions[:, None] * freqs[None, :]  # (K, half_d)
        emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)  # (K, D)
        # Shift so that the t=0 embedding is zero
        emb = emb - emb[0:1, :]
        return emb


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    depth: int
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    # ── MEM video-encoder parameters ──────────────────────────────────────────
    # Total number of frames fed to the encoder (K=1 → standard single-frame ViT).
    num_timesteps: int = 1
    # Insert a SpaceTimeSeparableBlock every N layers (only used when num_timesteps > 1).
    temporal_attn_every: int = 4
    # Drop history tokens from this layer onward (negative = count from end).
    # E.g. -4 means the last 4 layers operate only on the current frame's tokens.
    drop_history_after_layer: int = -4

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """Args:
            x: (B, N, D) when num_timesteps=1  OR  (B*K, N, D) when num_timesteps>1.
        Returns:
            x: (B, N, D)  – history tokens are dropped in upper layers.
        """
        out = {}
        K = self.num_timesteps
        use_video = K > 1

        if self.scan and not use_video:
            # ── Original scan path (π0.5 compatible) ──────────────────────────
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),  # 0=self, 2=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic)
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            # ── Layer-by-layer path (supports video encoder) ───────────────────
            drop_after = (
                self.depth + self.drop_history_after_layer
                if self.drop_history_after_layer < 0
                else self.drop_history_after_layer
            )

            for lyr in range(self.depth):
                use_temporal = use_video and (lyr % self.temporal_attn_every == 0)

                if use_temporal:
                    block_cur = SpaceTimeSeparableBlock(
                        name=f"encoderblock_{lyr}",
                        dtype_mm=self.dtype_mm,
                        mlp_dim=self.mlp_dim,
                        num_heads=self.num_heads,
                        dropout=self.dropout,
                        num_timesteps=K,
                    )
                else:
                    block_cur = Encoder1DBlock(
                        name=f"encoderblock_{lyr}",
                        dtype_mm=self.dtype_mm,
                        mlp_dim=self.mlp_dim,
                        num_heads=self.num_heads,
                        dropout=self.dropout,
                    )

                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)

                # After `drop_after` layer: discard history frame tokens,
                # keep only the current (last) frame.
                if use_video and lyr == drop_after:
                    BK, N, D = x.shape
                    B = BK // K
                    x = x.reshape(B, K, N, D)[:, -1, :, :]  # (B, N, D)
                    use_video = False  # subsequent layers work on B frames only

            if not (self.scan and not (K > 1)):
                out["pre_ln"] = x

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape  # n,l,d
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype_mm)(y)
        return x[:, 0]


class _Module(nn.Module):
    """ViT model."""

    num_classes: int | None = None
    patch_size: Sequence[int] = (16, 16)
    width: int = 768
    depth: int = 12
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    posemb: str = "learn"  # Can also be "sincos2d"
    rep_size: int | bool = False
    dropout: float = 0.0
    pool_type: str = "gap"  # Can also be "map" or "tok"
    head_zeroinit: bool = True
    scan: bool = False
    # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    # ── MEM video-encoder parameters (forwarded to Encoder) ──────────────────
    num_timesteps: int = 1       # K; =1 → single-frame ViT (π0.5 compatible)
    temporal_attn_every: int = 4
    drop_history_after_layer: int = -4

    @nn.compact
    def __call__(self, image, *, train=False):
        out = {}

        # Kevin edit: do patch extraction and posemb in float32,
        # because I feel like it's a bit safer.
        image = jnp.asarray(image, jnp.float32)

        # Patch extraction
        x = out["stem"] = nn.Conv(
            self.width,
            self.patch_size,
            strides=self.patch_size,
            padding="VALID",
            name="embedding",
            dtype=jnp.float32,
        )(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # Add posemb before adding extra token.
        x = out["with_posemb"] = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        n, _, c = x.shape  # n,l,d
        x = nn.Dropout(rate=self.dropout)(x, not train)

        # Kevin edit: now cast back to dtype_mm (potentially half precision)
        x = x.astype(self.dtype_mm)

        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            num_timesteps=self.num_timesteps,
            temporal_attn_every=self.temporal_attn_every,
            drop_history_after_layer=self.drop_history_after_layer,
            name="Transformer",
        )(x, deterministic=not train)
        encoded = out["encoded"] = x

        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype_mm,
            )(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]
            encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            pass
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        x_2d = jnp.reshape(encoded, [n, h, w, -1])

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            # NOTE: In the past we did not include tanh in pre_logits.
            # For few-shot, it should not matter much, as it whitens anyways.
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        return x, out


def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """Factory function, because linen really don't like what I'm doing!"""
    return _Module(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],
        # pylint:enable=line-too-long
        **patch,
    }
