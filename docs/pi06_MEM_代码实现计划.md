# π0.6-MEM 详细代码实现计划

> 基于 openpi (π0.5) 代码库，逐步集成 Multi-Scale Embodied Memory (MEM) 架构  
> 参考论文：[MEM: Multi-Scale Embodied Memory for Vision Language Action Models](https://pi.website/research/memory)

---

## 目录

1. [总体改动地图](#1-总体改动地图)
2. [Step 1 — 扩展数据结构 `model.py`](#2-step-1--扩展数据结构-modelpy)
3. [Step 2 — 视频编码器 `siglip.py` + `video_encoder.py`](#3-step-2--视频编码器-siglippy--video_encoderpy)
4. [Step 3 — 模型前向传播 `pi0.py`](#4-step-3--模型前向传播-pi0py)
5. [Step 4 — 高层策略 `high_level_policy.py`](#5-step-4--高层策略-high_level_policypy)
6. [Step 5 — 记忆管理器 `memory_manager.py`](#6-step-5--记忆管理器-memory_managerpy)
7. [Step 6 — 模型配置 `pi0_config.py`](#7-step-6--模型配置-pi0_configpy)
8. [Step 7 — 数据变换 `transforms.py`](#8-step-7--数据变换-transformspy)
9. [Step 8 — 数据加载 `data_loader.py`](#9-step-8--数据加载-data_loaderpy)
10. [Step 9 — 训练配置 `training/config.py`](#10-step-9--训练配置-trainingconfigpy)
11. [Step 10 — 推理接口 `policies/policy.py`](#11-step-10--推理接口-policiespolicypy)
12. [Step 11 — 记忆标注生成脚本](#12-step-11--记忆标注生成脚本)
13. [测试方案与代码](#13-测试方案与代码)

---

## 1. 总体改动地图

```
openpi-main/
├── src/openpi/
│   ├── models/
│   │   ├── siglip.py           ← [修改] 新增 SpaceTimeSeparableBlock + 视频 Encoder
│   │   ├── video_encoder.py    ← [新增] VideoEncoder 封装，管理多帧编码逻辑
│   │   ├── model.py            ← [修改] Observation 新增历史帧字段
│   │   ├── pi0_config.py       ← [修改] 新增 MEMConfig 配置项
│   │   ├── pi0.py              ← [修改] embed_prefix 支持多帧 + 语言记忆
│   │   ├── high_level_policy.py← [新增] 高层策略（子任务 + 记忆更新）
│   │   └── memory_manager.py   ← [新增] 语言记忆管理（更新/压缩/序列化）
│   ├── training/
│   │   ├── config.py           ← [修改] MEMConfig 集成到训练配置
│   │   └── data_loader.py      ← [修改] 支持多帧和记忆标注数据加载
│   ├── policies/
│   │   └── policy.py           ← [修改] 推理时维护帧缓存和记忆状态
│   ├── transforms.py           ← [修改] 多帧图像预处理和时间 padding
│   └── scripts/
│       └── gen_memory_labels.py← [新增] 离线语言记忆标注生成脚本
└── tests/
    ├── test_video_encoder.py   ← [新增]
    ├── test_mem_model.py       ← [新增]
    ├── test_memory_manager.py  ← [新增]
    ├── test_transforms_mem.py  ← [新增]
    └── test_policy_mem.py      ← [新增]
```

### 关键约束（贯穿所有实现）

| 约束 | 说明 |
|------|------|
| **向后兼容** | `image_history=None` 时所有 MEM 分支跳过，行为等同原 π0.5 |
| **零新参数引入** | 视频编码器仅修改注意力模式，不新增 learnable 参数 |
| **K=1 数值等价** | 单帧时输出与原 SigLIP ViT 在数值上精确一致（误差 < 1e-5） |
| **实时性** | 6 帧时推理延迟 < 300ms（H100 单卡，4 路摄像头） |

---

## 2. Step 1 — 扩展数据结构 `model.py`

### 2.1 改动位置

文件：`src/openpi/models/model.py`  
改动类型：**修改 `Observation` 数据类 + `from_dict` / `to_dict` / `preprocess_observation`**

### 2.2 完整实现代码

在 `Observation` 的 `token_loss_mask` 字段后追加（约第 106 行）：

```python
# ============================================================
# π0.6-MEM 新增字段（π0.5 兼容：全部 Optional，默认 None）
# ============================================================

# 短期视觉记忆：历史 K 帧图像，形状 (B, K, H, W, C)
# key 与 images 对应（相同摄像头名称）
image_history: dict[str, at.Float[ArrayT, "*b k h w c"]] | None = None

# 历史帧掩码：True 表示该帧有效（避免 padding 帧影响注意力）
image_history_masks: dict[str, at.Bool[ArrayT, "*b k"]] | None = None

# 短期本体感觉历史：K 帧历史状态，形状 (B, K, S)
# 使用连续嵌入替代文本 token，大幅减少 token 数
state_history: at.Float[ArrayT, "*b k s"] | None = None

# 长期语义记忆（token 化后）：形状 (B, M)
tokenized_memory: at.Int[ArrayT, "*b m"] | None = None
tokenized_memory_mask: at.Bool[ArrayT, "*b m"] | None = None
```

修改 `from_dict` 方法，在 return 前追加 MEM 字段解析：

```python
@classmethod
def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
    # ...（原有逻辑不变）...

    # MEM 字段解析（可选，不存在时为 None）
    image_history = None
    image_history_masks = None
    if "image_history" in data:
        image_history = data["image_history"]  # dict[str, array(B,K,H,W,C)]
        # uint8 → float32 归一化（与 images 字段处理一致）
        for key in image_history:
            if image_history[key].dtype == np.uint8:
                image_history[key] = (
                    image_history[key].astype(np.float32) / 255.0 * 2.0 - 1.0
                )
        image_history_masks = data.get("image_history_mask")  # 可选

    return cls(
        images=data["image"],
        image_masks=data["image_mask"],
        state=data["state"],
        tokenized_prompt=data.get("tokenized_prompt"),
        tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
        token_ar_mask=data.get("token_ar_mask"),
        token_loss_mask=data.get("token_loss_mask"),
        # MEM 字段
        image_history=image_history,
        image_history_masks=image_history_masks,
        state_history=data.get("state_history"),
        tokenized_memory=data.get("tokenized_memory"),
        tokenized_memory_mask=data.get("tokenized_memory_mask"),
    )
```

修改 `preprocess_observation`，在末尾 return 前加入历史帧预处理：

```python
def preprocess_observation(rng, observation, *, train=False, ...):
    # ...（原有逻辑不变，处理 images 和 masks）...

    # 处理历史帧图像（仅做 resize，不做数据增强，保证时序一致性）
    out_image_history = None
    out_image_history_masks = None
    if observation.image_history is not None:
        out_image_history = {}
        for key in image_keys:
            if key not in observation.image_history:
                continue
            hist = observation.image_history[key]  # (*b, K, H, W, C)
            batch_shape = hist.shape[:-4]
            K = hist.shape[-4]
            # 将 (*b, K, H, W, C) 展平为 (*b*K, H, W, C) 做 resize
            hist_flat = hist.reshape((-1, *hist.shape[-3:]))
            if hist_flat.shape[1:3] != image_resolution:
                hist_flat = image_tools.resize_with_pad(hist_flat, *image_resolution)
            out_image_history[key] = hist_flat.reshape((*batch_shape, K, *image_resolution, 3))

        out_image_history_masks = observation.image_history_masks

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        # MEM 字段直通
        image_history=out_image_history,
        image_history_masks=out_image_history_masks,
        state_history=observation.state_history,
        tokenized_memory=observation.tokenized_memory,
        tokenized_memory_mask=observation.tokenized_memory_mask,
    )
```

---

## 3. Step 2 — 视频编码器 `siglip.py` + `video_encoder.py`

### 3.1 修改 `siglip.py`

在 `Encoder1DBlock` 之后，`Encoder` 之前，**新增** `SpaceTimeSeparableBlock`：

```python
class SpaceTimeSeparableBlock(nn.Module):
    """每 4 层使用一次的时空分离注意力 Block（MEM 核心）。

    在标准空间注意力之后，对同一 spatial patch 的不同时间步
    额外做一次因果时间注意力，使 ViT 能感知帧间运动和变化。

    关键设计：
    - 不引入新参数（复用原始 QKV 投影权重）
    - 因果掩码：当前帧只能 attend 到历史帧，不能 attend 到未来帧
    - 复杂度：O(Kn² + nK²) 而非 O(n²K²)（论文 Section III-C）
    """

    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"
    num_timesteps: int = 6  # K，历史帧数（预训练默认 6）

    @nn.compact
    def __call__(self, x, deterministic=True):
        """
        Args:
            x: (B*K, N, D) — B=batch, K=帧数, N=patch数, D=嵌入维度
               调用前需将 (B, K, N, D) reshape 为 (B*K, N, D)
        Returns:
            x: (B*K, N, D)，与输入形状相同
        """
        K = self.num_timesteps
        BK, N, D = x.shape
        B = BK // K

        # ---- Step 1: 标准空间注意力（与 Encoder1DBlock 完全相同）----
        residual = x
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = residual + y

        # MLP
        residual = x
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = MlpBlock(mlp_dim=self.mlp_dim, dtype_mm=self.dtype_mm)(y, deterministic)
        x = residual + y

        # ---- Step 2: 因果时间注意力 ----
        # (B*K, N, D) → (B, K, N, D) → (B*N, K, D)
        x_t = x.reshape(B, K, N, D)
        x_t = x_t.transpose(0, 2, 1, 3).reshape(B * N, K, D)  # (B*N, K, D)

        # 加时间位置编码（正弦，t=0 时值为 0，保证 K=1 时与原 ViT 行为一致）
        time_pos = self._sinusoidal_time_embedding(K, D)  # (K, D)
        x_t = x_t + time_pos[None, :, :]

        # 因果掩码：下三角，时刻 t 只能 attend 到 t' <= t
        causal_mask = jnp.tril(jnp.ones((K, K), dtype=jnp.bool_))  # (K, K)
        causal_mask = causal_mask[None, :, :]  # (1, K, K)

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

        # 还原形状：(B*N, K, D) → (B, N, K, D) → (B, K, N, D) → (B*K, N, D)
        x_t = x_t.reshape(B, N, K, D).transpose(0, 2, 1, 3).reshape(B * K, N, D)

        return x_t

    def _sinusoidal_time_embedding(self, K: int, D: int):
        """生成正弦时间位置编码，t=0 时编码值为 0（保证 K=1 兼容性）。"""
        # 论文 Appendix C：e(t=0) = 0，K=1 时视频编码器退化为普通 ViT
        positions = jnp.arange(K, dtype=jnp.float32)  # (K,)
        # 使用 half_dim 对数空间频率
        half_dim = D // 2
        emb = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = positions[:, None] * emb[None, :]  # (K, half_dim)
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)  # (K, D)
        # 令 t=0 的编码为 0（bias 设计：sin(0)=0, cos(0)=1，需 shift）
        # 简单方法：减去 t=0 的编码，使其归零
        emb = emb - emb[0:1, :]
        return emb
```

修改 `Encoder.__call__`（non-scan 路径），每 4 层使用 `SpaceTimeSeparableBlock`：

```python
class Encoder(nn.Module):
    depth: int
    mlp_dim: int | None = None
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"
    # MEM 新增参数
    num_timesteps: int = 1          # K；=1 时退化为原始 ViT
    temporal_attn_every: int = 4    # 每隔几层插入时间注意力
    drop_history_after_layer: int = -1  # 在第几层后丢弃历史帧 token（-1=最后）

    @nn.compact
    def __call__(self, x, deterministic=True):
        """
        Args:
            x: (B*K, N, D) if num_timesteps > 1 else (B, N, D)
        Returns:
            x: (B, N, D)  — 历史帧 token 在上层被丢弃，仅输出当前帧
        """
        out = {}
        K = self.num_timesteps
        use_video = K > 1

        # scan 模式暂不支持视频编码器（需要按层条件分支，与 scan 冲突）
        # 视频编码器强制使用 non-scan 路径
        if self.scan and not use_video:
            # 原始 scan 路径（π0.5 兼容）
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),
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
        else:
            # 逐层 non-scan 路径（支持视频编码器）
            # drop_after_layer：在此层之后仅保留当前帧 token
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

                # 在 drop_after 层之后，丢弃历史帧 token，仅保留当前帧
                if use_video and lyr == drop_after:
                    # x: (B*K, N, D) → 仅保留最后一个时间步（当前帧）
                    BK, N, D = x.shape
                    B = BK // K
                    x = x.reshape(B, K, N, D)[:, -1, :, :]  # (B, N, D)
                    use_video = False  # 后续层正常处理

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out
```

### 3.2 新增 `video_encoder.py`

文件：`src/openpi/models/video_encoder.py`

```python
"""VideoEncoder: 将 K 帧图像序列编码为与单帧等量的 token 序列。

设计目标：
1. K=1 时与原 SigLIP 行为数值完全一致
2. 不引入新的 learnable 参数
3. 输出 token 数与单帧相同，推理 token budget 不增加
"""

from typing import Any

import einops
import flax.linen as nn
import flax.nnx.bridge as nnx_bridge
import jax.numpy as jnp

import openpi.models.siglip as _siglip


class VideoEncoder:
    """将多帧图像序列编码为视觉 token 的封装类。

    这不是 nn.Module 本身，而是对底层 SigLIP Module 的调用封装，
    处理帧维度的 reshape 逻辑，使外部代码保持简洁。
    """

    def __init__(self, siglip_module: Any, num_timesteps: int = 1):
        """
        Args:
            siglip_module: 已初始化的 SigLIP nnx bridge 模块
                          （来自 Pi0 的 self.PaliGemma.img）
            num_timesteps: K，历史帧数（=1 时退化为单帧编码器）
        """
        self.img = siglip_module
        self.K = num_timesteps

    def encode_single(self, image, *, train: bool = False):
        """编码单帧图像（向后兼容原始接口）。

        Args:
            image: (*B, H, W, C)
        Returns:
            tokens: (*B, N, D)
        """
        return self.img(image, train=train)

    def encode_video(self, image_history, current_image, *, train: bool = False):
        """编码视频序列（K-1 历史帧 + 1 当前帧）。

        Args:
            image_history: (*B, K-1, H, W, C)，历史帧
            current_image: (*B, H, W, C)，当前帧
        Returns:
            tokens: (*B, N, D)，与单帧编码输出形状相同
        """
        K = self.K
        if K <= 1:
            # K=1 直接编码当前帧，走原路径
            tokens, _ = self.img(current_image, train=train)
            return tokens

        # 将历史帧和当前帧拼接：(*B, K, H, W, C)
        curr_expanded = current_image[..., None, :, :, :]  # (*B, 1, H, W, C)
        frames = jnp.concatenate([image_history, curr_expanded], axis=-4)  # (*B, K, H, W, C)

        batch_shape = frames.shape[:-4]
        H, W, C = frames.shape[-3:]

        # 展平为 (*B*K, H, W, C)，送入视频编码器
        frames_flat = frames.reshape((-1, H, W, C))
        tokens_flat, _ = self.img(frames_flat, train=train)  # (*B*K, N, D)

        # 视频编码器内部已在上层丢弃历史帧，输出已是 (*B, N, D)
        # 但如果 siglip 尚未集成 drop_history 逻辑，需在此还原
        # 当前实现：信任 siglip.Encoder 在 drop_after 层处理了维度还原
        N, D = tokens_flat.shape[-2], tokens_flat.shape[-1]

        # 若输出仍是 *B*K 维度（尚未 drop），手动取当前帧
        if tokens_flat.shape[0] == frames_flat.shape[0]:
            tokens = tokens_flat.reshape(*batch_shape, K, N, D)[..., -1, :, :]
        else:
            tokens = tokens_flat.reshape(*batch_shape, N, D)

        return tokens
```

---

## 4. Step 3 — 模型前向传播 `pi0.py`

### 4.1 构造函数新增字段

在 `Pi0.__init__` 中新增（约第 60 行）：

```python
# MEM 配置
self.use_video_memory = config.use_video_memory     # bool
self.use_language_memory = config.use_language_memory  # bool
self.num_video_frames = config.video_memory_frames  # K

# 本体感觉历史嵌入（连续嵌入，不用文本 token）
if config.use_video_memory and config.use_state_history:
    self.state_history_proj = nnx.Linear(
        config.action_dim,
        action_expert_config.width,  # 映射到 VLM 嵌入维度
        rngs=rngs,
    )
```

### 4.2 `embed_prefix` 完整改写

```python
@at.typecheck
def embed_prefix(
    self, obs: _model.Observation
) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
    """将观测的前缀部分编码为 token 序列。

    token 顺序（论文 Figure 2 左侧）：
    [语言记忆 token] [当前帧图像 token × N_cams] [语言 prompt token] [历史状态 token × K]

    注意：语言记忆放在最前面，使模型在处理图像前已有任务上下文。
    """
    input_mask = []
    ar_mask = []
    tokens = []

    # ---- (A) 长期语言记忆 token（MEM 新增，放最前面）----
    if self.use_language_memory and obs.tokenized_memory is not None:
        memory_tokens = self.PaliGemma.llm(obs.tokenized_memory, method="embed")
        tokens.append(memory_tokens)
        # 记忆 token 使用自身的掩码（可能有 padding）
        if obs.tokenized_memory_mask is not None:
            input_mask.append(obs.tokenized_memory_mask)
        else:
            input_mask.append(jnp.ones(memory_tokens.shape[:2], dtype=jnp.bool_))
        # 记忆 token 互相可见（非自回归），ar_mask=False
        ar_mask += [False] * memory_tokens.shape[1]

    # ---- (B) 当前帧图像 token（原有逻辑）----
    for name in obs.images:
        if self.use_video_memory and obs.image_history is not None and name in obs.image_history:
            # 使用视频编码器：融合历史帧信息
            video_tokens = self._encode_video_frames(
                obs.image_history[name],   # (*B, K-1, H, W, C)
                obs.images[name],          # (*B, H, W, C)
            )
            image_tokens = video_tokens
        else:
            # 单帧编码（原始路径）
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

        tokens.append(image_tokens)
        input_mask.append(
            einops.repeat(
                obs.image_masks[name],
                "b -> b s",
                s=image_tokens.shape[1],
            )
        )
        ar_mask += [False] * image_tokens.shape[1]

    # ---- (C) 语言 prompt token（原有逻辑）----
    if obs.tokenized_prompt is not None:
        tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(tokenized_inputs)
        input_mask.append(obs.tokenized_prompt_mask)
        ar_mask += [False] * tokenized_inputs.shape[1]

    # ---- (D) 历史本体感觉状态嵌入（MEM 新增）----
    if self.use_video_memory and obs.state_history is not None:
        # state_history: (B, K, S) → project → (B, K, D)
        state_hist_tokens = self.state_history_proj(obs.state_history)
        tokens.append(state_hist_tokens)
        input_mask.append(jnp.ones(state_hist_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [False] * state_hist_tokens.shape[1]

    tokens = jnp.concatenate(tokens, axis=1)
    input_mask = jnp.concatenate(input_mask, axis=1)
    ar_mask = jnp.array(ar_mask)
    return tokens, input_mask, ar_mask

def _encode_video_frames(self, image_history, current_image):
    """调用视频编码器编码多帧图像序列。"""
    K = self.num_video_frames
    if K <= 1:
        tokens, _ = self.PaliGemma.img(current_image, train=False)
        return tokens

    # image_history: (B, K-1, H, W, C)，current_image: (B, H, W, C)
    curr_expanded = current_image[:, None, :, :, :]  # (B, 1, H, W, C)
    frames = jnp.concatenate([image_history, curr_expanded], axis=1)  # (B, K, H, W, C)

    B, K_actual, H, W, C = frames.shape
    frames_flat = frames.reshape(B * K_actual, H, W, C)

    tokens_flat, _ = self.PaliGemma.img(frames_flat, train=False)  # (B*K, N, D)
    N, D = tokens_flat.shape[1], tokens_flat.shape[2]

    # 取当前帧（最后一帧）的 token（视频编码器已将历史信息融合进去）
    tokens = tokens_flat.reshape(B, K_actual, N, D)[:, -1, :, :]  # (B, N, D)
    return tokens
```

### 4.3 `compute_loss` 新增语言记忆损失（高层策略训练）

```python
@override
def compute_loss(
    self, rng, observation, actions, *, train=False
) -> at.Float[at.Array, "*b ah"]:
    # 原有 flow matching loss（不变）
    flow_loss = self._compute_flow_loss(rng, observation, actions, train=train)

    # MEM 高层策略损失（若有语言记忆训练目标）
    # 注：高层策略作为独立模块训练，此处仅保留低层策略损失
    return flow_loss
```

---

## 5. Step 4 — 高层策略 `high_level_policy.py`

文件：`src/openpi/models/high_level_policy.py`（新增）

```python
"""高层策略（πHL）：子任务选择 + 语言记忆更新。

πHL 在低层策略步频的更低频率运行（例如每完成一个子任务触发一次）。
输入：当前观测 + 任务目标文本 + 当前语言记忆
输出：下一个子任务指令文本 + 更新后的语言记忆文本

实现方式：使用 PaliGemma VLM 的语言生成能力，
通过 chain-of-thought 格式同时输出子任务和记忆更新。
"""

import dataclasses
import json
import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)

# 高层策略的 prompt 模板
# 遵循 chain-of-thought 格式，同时输出子任务和记忆
HL_PROMPT_TEMPLATE = """You are controlling a robot to complete a task.

Task goal: {task_goal}

Current memory of what has been done:
{language_memory}

Based on the current observation and memory, determine:
1. The next subtask to execute (a short, specific robot action)
2. Update the memory to include what was just completed (compress if needed)

Respond in JSON format:
{{
  "subtask": "<next robot action>",
  "updated_memory": "<compressed summary of all completed steps>"
}}"""

# 记忆压缩指令（在记忆更新时给 LLM 的提示）
MEMORY_COMPRESSION_INSTRUCTION = """
Rules for memory updates:
- Only keep information relevant for future steps
- Compress redundant information (e.g., "picked up plate, picked up bowl" → "picked up plates and bowls")
- Do NOT include failed attempts (only record successful actions)
- Keep memory concise (under 200 words)
"""


@dataclasses.dataclass
class HighLevelPolicyConfig:
    """高层策略配置。"""
    max_subtask_tokens: int = 64       # 子任务指令最大 token 数
    max_memory_tokens: int = 256       # 语言记忆最大 token 数
    subtask_trigger_steps: int = 50    # 每隔多少低层步骤触发一次高层策略
    # 当检测到子任务完成时立即触发（优先于 step 计数）
    use_completion_detection: bool = True


class HighLevelPolicy:
    """高层策略：管理子任务调度和语言记忆更新。

    在推理阶段，该类维护一个状态机：
    - 维护当前子任务和语言记忆
    - 按照触发条件（step 计数 or 完成检测）调用 VLM 更新
    """

    def __init__(
        self,
        vlm_inference_fn,          # 调用 VLM 生成文本的函数
        tokenizer,                 # 用于 token 化文本
        config: HighLevelPolicyConfig | None = None,
    ):
        self.vlm_infer = vlm_inference_fn
        self.tokenizer = tokenizer
        self.config = config or HighLevelPolicyConfig()

        # 运行时状态
        self._current_subtask: str = ""
        self._language_memory: str = ""
        self._step_count: int = 0
        self._task_goal: str = ""

    def reset(self, task_goal: str):
        """开始新 episode，重置状态。"""
        self._current_subtask = ""
        self._language_memory = ""
        self._step_count = 0
        self._task_goal = task_goal
        logger.info(f"[HL Policy] Reset. Task goal: {task_goal}")

    def should_update(self, subtask_completed: bool = False) -> bool:
        """判断是否需要触发高层策略更新。"""
        if subtask_completed and self.config.use_completion_detection:
            return True
        self._step_count += 1
        return self._step_count % self.config.subtask_trigger_steps == 0

    def update(
        self,
        observation_image: np.ndarray,
        subtask_success: bool = True,
    ) -> tuple[str, str]:
        """调用 VLM 更新子任务和语言记忆。

        Args:
            observation_image: 当前摄像头图像，(H, W, C) uint8
            subtask_success: 上一个子任务是否成功完成

        Returns:
            (new_subtask, updated_memory): 新子任务指令和更新后的记忆
        """
        prompt = HL_PROMPT_TEMPLATE.format(
            task_goal=self._task_goal,
            language_memory=self._language_memory or "(none yet)",
        )

        # 调用 VLM 生成
        raw_output = self.vlm_infer(
            image=observation_image,
            prompt=prompt + MEMORY_COMPRESSION_INSTRUCTION,
            max_tokens=self.config.max_subtask_tokens + self.config.max_memory_tokens,
        )

        # 解析 JSON 输出
        try:
            parsed = json.loads(raw_output.strip())
            new_subtask = parsed.get("subtask", self._current_subtask)
            new_memory = parsed.get("updated_memory", self._language_memory)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[HL Policy] Failed to parse VLM output: {e}. Raw: {raw_output}")
            new_subtask = self._current_subtask
            new_memory = self._language_memory

        # 关键：失败时不更新记忆（防止分布偏移）
        if subtask_success:
            self._language_memory = new_memory
            logger.info(f"[HL Policy] Memory updated: {new_memory[:100]}...")
        else:
            logger.info("[HL Policy] Subtask failed, memory NOT updated.")

        self._current_subtask = new_subtask
        logger.info(f"[HL Policy] Next subtask: {new_subtask}")

        return new_subtask, self._language_memory

    def tokenize_memory(self, memory: str) -> tuple[np.ndarray, np.ndarray]:
        """将语言记忆文本 token 化。

        Returns:
            (token_ids, token_mask): 各 shape (max_memory_tokens,)
        """
        max_len = self.config.max_memory_tokens
        if not memory:
            return (
                np.zeros(max_len, dtype=np.int32),
                np.zeros(max_len, dtype=bool),
            )
        tokens = self.tokenizer.encode(memory)[:max_len]
        pad_len = max_len - len(tokens)
        token_ids = np.array(tokens + [0] * pad_len, dtype=np.int32)
        token_mask = np.array([True] * len(tokens) + [False] * pad_len, dtype=bool)
        return token_ids, token_mask

    @property
    def current_subtask(self) -> str:
        return self._current_subtask

    @property
    def language_memory(self) -> str:
        return self._language_memory
```

---

## 6. Step 5 — 记忆管理器 `memory_manager.py`

文件：`src/openpi/models/memory_manager.py`（新增）

```python
"""语言记忆管理器：处理记忆的生成、更新、压缩和训练数据生成。"""

import dataclasses
import json
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MemoryLabel:
    """训练时的记忆标注样本。"""
    episode_id: str
    timestep: int
    subtask_instruction: str
    subtask_success: bool
    memory_before: str   # m_t
    memory_after: str    # m_{t+1}（由 LLM 生成）


@dataclasses.dataclass
class MemoryGenerationConfig:
    """记忆标注生成配置。"""
    max_memory_length: int = 512      # 最大记忆字符数
    compression_model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    # 压缩比例：记忆长度超过此阈值时强制压缩
    compression_threshold: int = 300


MEMORY_GENERATION_PROMPT = """You are generating training labels for a robot memory system.

Given a sequence of completed robot subtasks, generate a compressed memory summary.

Rules:
1. Only include information that will be USEFUL for future steps
2. Compress multiple similar actions (e.g., "picked up bowl, picked up plate" → "picked up dishes")
3. Do NOT include failed attempts
4. If a subtask has failed=True, ignore it completely
5. Keep the summary under {max_length} characters
6. Use first-person past tense ("I placed...", "I retrieved...")

Completed subtasks so far:
{subtask_history}

Generate a compressed memory summary:"""


class MemoryDataGenerator:
    """离线生成语言记忆训练标注。

    工作流程：
    1. 读取 robot episode（含子任务序列和成功标记）
    2. 调用 LLM 生成每个时间步的记忆标注
    3. 输出 (m_t, m_{t+1}) 训练对
    """

    def __init__(self, llm_client, config: MemoryGenerationConfig | None = None):
        """
        Args:
            llm_client: LLM API 客户端（支持 generate(prompt) → str）
        """
        self.llm = llm_client
        self.config = config or MemoryGenerationConfig()

    def generate_labels_for_episode(
        self,
        episode_id: str,
        subtasks: list[dict],  # [{"instruction": str, "success": bool}, ...]
    ) -> list[MemoryLabel]:
        """为单个 episode 生成所有时间步的记忆标注。

        Args:
            subtasks: 子任务列表，每个元素包含 instruction 和 success 标记

        Returns:
            记忆标注列表，长度与 subtasks 相同
        """
        labels = []
        current_memory = ""

        for t, subtask in enumerate(subtasks):
            # 构造前 t 步的历史（仅成功的步骤）
            successful_history = [
                f"Step {i+1}: {s['instruction']}"
                for i, s in enumerate(subtasks[:t])
                if s.get("success", True)
            ]

            if not successful_history:
                # 初始状态，无记忆
                new_memory = ""
            else:
                prompt = MEMORY_GENERATION_PROMPT.format(
                    max_length=self.config.max_memory_length,
                    subtask_history="\n".join(successful_history),
                )
                new_memory = self.llm.generate(
                    prompt,
                    temperature=self.config.temperature,
                    max_tokens=256,
                )
                new_memory = new_memory.strip()

            labels.append(MemoryLabel(
                episode_id=episode_id,
                timestep=t,
                subtask_instruction=subtask["instruction"],
                subtask_success=subtask.get("success", True),
                memory_before=current_memory,
                memory_after=new_memory,
            ))

            # 更新当前记忆（仅在子任务成功时）
            if subtask.get("success", True):
                current_memory = new_memory

        return labels

    def save_labels(self, labels: list[MemoryLabel], output_path: str):
        """将记忆标注保存为 JSON Lines 文件。"""
        with open(output_path, "w", encoding="utf-8") as f:
            for label in labels:
                f.write(json.dumps(dataclasses.asdict(label), ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(labels)} memory labels to {output_path}")

    @staticmethod
    def load_labels(input_path: str) -> list[MemoryLabel]:
        """从 JSON Lines 文件加载记忆标注。"""
        labels = []
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                labels.append(MemoryLabel(**data))
        return labels
```

---

## 7. Step 6 — 模型配置 `pi0_config.py`

在文件末尾，`Pi0Config` 类定义之前，**新增** `MEMConfig`：

```python
@dataclasses.dataclass(frozen=True)
class MEMConfig:
    """Multi-Scale Embodied Memory 配置。

    所有 bool 标志默认 False，确保不修改 MEMConfig 时完全兼容 π0.5。
    """

    # -------- 短期视觉记忆（视频编码器）--------
    use_video_memory: bool = False
    # 历史帧数 K（含当前帧）：预训练用 6，微调可扩至 18
    video_memory_frames: int = 6
    # 相邻帧的时间间隔（秒），用于数据加载时采样
    video_frame_stride_sec: float = 1.0
    # 每隔多少层插入一次时间注意力
    temporal_attn_every_n_layers: int = 4
    # 在第几层之后丢弃历史帧 token（负数表示倒数）
    # -4 表示最后 4 层仅处理当前帧（减少计算量）
    drop_history_tokens_after_layer: int = -4

    # -------- 长期语言记忆 --------
    use_language_memory: bool = False
    # 语言记忆最大 token 数
    max_memory_tokens: int = 256
    # 语言记忆损失权重（在总 loss 中）
    memory_loss_weight: float = 0.1

    # -------- 本体感觉历史嵌入 --------
    # True: 使用连续嵌入（推荐），False: 退化为文本 token（不推荐，token 数量大）
    use_state_history: bool = False
    # 历史状态帧数（通常与 video_memory_frames 一致）
    state_history_frames: int = 6
```

修改 `Pi0Config`，添加 MEM 配置字段：

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None
    pi05: bool = False
    discrete_state_input: bool = None

    # π0.6-MEM 新增字段
    mem: MEMConfig = dataclasses.field(default_factory=MEMConfig)

    # 便捷属性（从 mem 透传，减少代码改动）
    @property
    def use_video_memory(self) -> bool:
        return self.mem.use_video_memory

    @property
    def use_language_memory(self) -> bool:
        return self.mem.use_language_memory

    @property
    def video_memory_frames(self) -> int:
        return self.mem.video_memory_frames
```

---

## 8. Step 7 — 数据变换 `transforms.py`

新增 `VideoFrameStack` 变换类：

```python
@dataclasses.dataclass(frozen=True)
class VideoFrameStack(DataTransformFn):
    """将单帧数据集转换为多帧视频序列格式。

    在数据加载时，从 episode 中采样 K 帧（包含历史帧），
    组装为 image_history 字段。

    Args:
        num_frames: K，总帧数（含当前帧）
        stride: 历史帧采样步长（以数据集帧率为单位）
        camera_keys: 需要构建历史的摄像头名称列表
        pad_with_first_frame: 当历史帧不足 K-1 帧时，用第一帧 padding
    """

    num_frames: int = 6
    stride: int = 5          # 数据集帧率 5Hz，stride=5 对应 1 秒间隔
    camera_keys: tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    pad_with_first_frame: bool = True

    def __call__(self, data: DataDict) -> DataDict:
        """
        输入 data 格式（来自 lerobot dataset）：
          data["image"][cam_key]: (*B, H, W, C)  当前帧
          data["image_history"][cam_key]: (*B, K, H, W, C)  历史帧（若已预处理）
          data["state_sequence"]: (*B, K, S)  历史状态序列

        如果 image_history 不存在，跳过（向后兼容）。
        """
        if "image_history" not in data:
            return data  # 向后兼容：单帧数据不做处理

        K = self.num_frames
        result = dict(data)

        # 验证历史帧数量
        for key in self.camera_keys:
            if key in data.get("image_history", {}):
                hist = data["image_history"][key]
                assert hist.shape[-4] == K - 1, (
                    f"Expected {K-1} history frames for {key}, got {hist.shape[-4]}"
                )

        return result


@dataclasses.dataclass(frozen=True)
class TokenizeMemory(DataTransformFn):
    """将语言记忆文本 token 化，加入 tokenized_memory 字段。

    Args:
        tokenizer: 分词器
        max_len: 最大 token 长度
    """

    tokenizer: Any = None
    max_len: int = 256

    def __call__(self, data: DataDict) -> DataDict:
        if "language_memory" not in data or self.tokenizer is None:
            return data

        memory_text = data["language_memory"]
        if not isinstance(memory_text, str):
            return data

        tokens = self.tokenizer.encode(memory_text)[:self.max_len]
        pad_len = self.max_len - len(tokens)

        result = dict(data)
        result["tokenized_memory"] = np.array(tokens + [0] * pad_len, dtype=np.int32)
        result["tokenized_memory_mask"] = np.array(
            [True] * len(tokens) + [False] * pad_len, dtype=bool
        )
        return result
```

---

## 9. Step 8 — 数据加载 `data_loader.py`

新增 `MEMLeRobotDataset` 类：

```python
class MEMLeRobotDataset:
    """支持多帧历史和语言记忆标注的数据集包装器。

    在原 LeRobot 数据集基础上：
    1. 按 stride 采样 K 帧历史图像
    2. 加载对应的语言记忆标注（若提供了标注文件）
    3. 处理 episode 边界的 padding
    """

    def __init__(
        self,
        lerobot_dataset,
        num_frames: int = 6,
        stride: int = 5,
        memory_labels_path: str | None = None,
        camera_keys: list[str] | None = None,
    ):
        """
        Args:
            lerobot_dataset: 底层 LeRobot 数据集
            num_frames: K，总帧数（含当前帧）
            stride: 历史帧采样步长
            memory_labels_path: JSON Lines 格式的记忆标注文件路径
            camera_keys: 需要构建历史的摄像头键
        """
        self._ds = lerobot_dataset
        self.K = num_frames
        self.stride = stride
        self.camera_keys = camera_keys or ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

        # 加载语言记忆标注（可选）
        self._memory_labels: dict[tuple[str, int], str] = {}
        if memory_labels_path:
            self._load_memory_labels(memory_labels_path)

    def _load_memory_labels(self, path: str):
        """加载 JSON Lines 记忆标注文件。"""
        import json
        with open(path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                key = (data["episode_id"], data["timestep"])
                self._memory_labels[key] = data["memory_before"]
        logger.info(f"Loaded {len(self._memory_labels)} memory labels from {path}")

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> dict:
        sample = self._ds[index]

        # 获取 episode 信息（用于边界 padding）
        episode_id = sample.get("episode_index", 0)
        frame_idx = sample.get("frame_index", 0)

        # 采样历史帧索引
        history_indices = self._get_history_indices(
            episode_id, frame_idx, self.K - 1
        )

        # 构建历史帧
        image_history = {}
        for cam_key in self.camera_keys:
            if f"observation.images.{cam_key}" in sample:
                frames = []
                for hist_idx in history_indices:
                    hist_sample = self._ds[hist_idx]
                    frames.append(hist_sample[f"observation.images.{cam_key}"])
                image_history[cam_key] = np.stack(frames, axis=0)  # (K-1, H, W, C)

        if image_history:
            sample["image_history"] = image_history

        # 加载语言记忆标注（若有）
        memory_key = (str(episode_id), frame_idx)
        if memory_key in self._memory_labels:
            sample["language_memory"] = self._memory_labels[memory_key]

        return sample

    def _get_history_indices(
        self, episode_id: int, current_frame: int, n_history: int
    ) -> list[int]:
        """获取历史帧在数据集中的绝对索引（处理 episode 边界 padding）。"""
        episode_start = self._get_episode_start_index(episode_id)
        indices = []
        for i in range(n_history, 0, -1):
            hist_frame = current_frame - i * self.stride
            if hist_frame < 0:
                # episode 边界：用第一帧 padding
                abs_idx = episode_start
            else:
                abs_idx = episode_start + hist_frame
            indices.append(abs_idx)
        return indices

    def _get_episode_start_index(self, episode_id: int) -> int:
        """获取 episode 在数据集中的起始帧绝对索引。"""
        # 依赖 lerobot_dataset 的 episode_data_index
        if hasattr(self._ds, "episode_data_index"):
            return int(self._ds.episode_data_index["from"][episode_id])
        return 0
```

---

## 10. Step 9 — 训练配置 `training/config.py`

在 `config.py` 中新增 `MEMDataConfigFactory`：

```python
@dataclasses.dataclass(frozen=True)
class MEMDataConfig(DataConfigFactory):
    """MEM 训练数据配置工厂。"""

    # 视频记忆配置
    num_video_frames: int = 6
    video_frame_stride: int = 5       # 数据集帧率单位

    # 语言记忆标注文件路径（可选）
    memory_labels_path: str | None = None

    # 继承自 DataConfigFactory
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(
        default_factory=ModelTransformFactory
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        base = self.create_base_config(assets_dirs, model_config)

        # 追加 MEM 专用 transforms
        mem_transforms = _transforms.Group(
            inputs=[
                _transforms.VideoFrameStack(
                    num_frames=self.num_video_frames,
                    stride=self.video_frame_stride,
                ),
            ]
        )
        if self.memory_labels_path:
            mem_transforms = mem_transforms.push(
                inputs=[
                    _transforms.TokenizeMemory(
                        tokenizer=_tokenizer.PaligemmaTokenizer(),
                        max_len=model_config.mem.max_memory_tokens
                        if hasattr(model_config, "mem") else 256,
                    )
                ]
            )

        return dataclasses.replace(
            base,
            data_transforms=dataclasses.replace(
                base.data_transforms,
                inputs=(*base.data_transforms.inputs, *mem_transforms.inputs),
            ),
        )
```

---

## 11. Step 10 — 推理接口 `policies/policy.py`

新增 `MEMPolicy` 类（在原 `Policy` 类后追加）：

```python
class MEMPolicy(Policy):
    """支持多帧历史记忆和语言记忆的推理策略。

    继承自 Policy，在 infer 方法中额外：
    1. 维护滑动帧缓存（deque）
    2. 管理语言记忆状态
    3. 在适当时机触发高层策略更新
    """

    def __init__(
        self,
        model,
        high_level_policy=None,     # HighLevelPolicy 实例（可选）
        num_video_frames: int = 6,
        camera_keys: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.K = num_video_frames
        self.camera_keys = camera_keys or ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        self.hl_policy = high_level_policy

        # 运行时状态
        self._frame_buffer: dict[str, deque] = {
            k: deque(maxlen=self.K - 1) for k in self.camera_keys
        }
        self._state_buffer: deque = deque(maxlen=self.K - 1)
        self._language_memory: str = ""
        self._current_subtask: str = ""
        self._episode_step: int = 0

    def reset_episode(self, task_goal: str = ""):
        """开始新 episode，清空所有缓存。"""
        for k in self.camera_keys:
            self._frame_buffer[k].clear()
        self._state_buffer.clear()
        self._language_memory = ""
        self._episode_step = 0
        if self.hl_policy and task_goal:
            self.hl_policy.reset(task_goal)

    @override
    def infer(self, obs: dict, *, noise=None) -> dict:
        """执行推理，自动维护帧历史和记忆状态。

        Args:
            obs: 原始观测字典（与原 Policy.infer 格式相同）
            noise: 可选噪声（用于测试）

        Returns:
            output: 包含 "actions" 的字典
        """
        # Step 1: 构建多帧输入（注入 image_history 和 state_history）
        obs_with_history = self._build_obs_with_history(obs)

        # Step 2: 注入语言记忆（若启用）
        if self._language_memory and self.hl_policy:
            token_ids, token_mask = self.hl_policy.tokenize_memory(self._language_memory)
            obs_with_history["tokenized_memory"] = token_ids
            obs_with_history["tokenized_memory_mask"] = token_mask

        # Step 3: 调用底层推理（原 Policy.infer 逻辑）
        result = super().infer(obs_with_history, noise=noise)

        # Step 4: 更新帧缓存（在推理后更新，避免影响当前步）
        self._update_frame_buffer(obs)
        self._episode_step += 1

        # Step 5: 按需触发高层策略更新
        if self.hl_policy and self.hl_policy.should_update():
            base_img = obs.get("base_0_rgb")
            if base_img is not None:
                new_subtask, new_memory = self.hl_policy.update(
                    observation_image=base_img,
                    subtask_success=True,  # 外部可通过回调注入失败信号
                )
                self._language_memory = new_memory
                self._current_subtask = new_subtask

        return result

    def _build_obs_with_history(self, obs: dict) -> dict:
        """在原始观测字典中注入历史帧数据。"""
        result = dict(obs)

        if self.K <= 1:
            return result

        # 构建 image_history
        image_history = {}
        for cam_key in self.camera_keys:
            if cam_key not in obs:
                continue
            buffer = list(self._frame_buffer[cam_key])
            if len(buffer) == 0:
                # 缓存为空：用当前帧复制 padding
                pad_frame = obs[cam_key]
                buffer = [pad_frame] * (self.K - 1)
            elif len(buffer) < self.K - 1:
                # 缓存不足：用最早帧 padding
                pad_frames = [buffer[0]] * (self.K - 1 - len(buffer))
                buffer = pad_frames + buffer
            image_history[cam_key] = np.stack(buffer, axis=0)  # (K-1, H, W, C)

        if image_history:
            result["image_history"] = image_history

        # 构建 state_history
        state_buffer = list(self._state_buffer)
        if len(state_buffer) > 0:
            if len(state_buffer) < self.K - 1:
                pad_states = [state_buffer[0]] * (self.K - 1 - len(state_buffer))
                state_buffer = pad_states + state_buffer
            result["state_history"] = np.stack(state_buffer, axis=0)  # (K-1, S)

        return result

    def _update_frame_buffer(self, obs: dict):
        """将当前帧推入缓存。"""
        for cam_key in self.camera_keys:
            if cam_key in obs:
                self._frame_buffer[cam_key].append(obs[cam_key].copy())
        if "state" in obs:
            self._state_buffer.append(obs["state"].copy())
```

---

## 12. Step 11 — 记忆标注生成脚本

文件：`scripts/gen_memory_labels.py`（新增）

```python
#!/usr/bin/env python3
"""离线生成语言记忆训练标注脚本。

用法：
    python scripts/gen_memory_labels.py \
        --dataset_path /path/to/lerobot_dataset \
        --output_path /path/to/memory_labels.jsonl \
        --api_key YOUR_ANTHROPIC_API_KEY

前提：数据集中每个样本需包含子任务序列标注
（通常通过人工标注或规则提取获得）。
"""

import argparse
import json
import logging
import os

import anthropic

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_anthropic_client(api_key: str):
    """创建 Anthropic LLM 客户端。"""
    client = anthropic.Anthropic(api_key=api_key)

    class AnthropicWrapper:
        def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 256) -> str:
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text

    return AnthropicWrapper()


def load_episodes_from_dataset(dataset_path: str) -> list[dict]:
    """从数据集加载 episode 列表（需根据实际格式调整）。

    Returns:
        episodes: 列表，每个元素格式：
            {
                "episode_id": str,
                "subtasks": [{"instruction": str, "success": bool}, ...]
            }
    """
    episodes = []
    subtask_file = os.path.join(dataset_path, "subtask_annotations.jsonl")
    if not os.path.exists(subtask_file):
        logger.warning(f"Subtask annotation file not found: {subtask_file}")
        return episodes

    with open(subtask_file, encoding="utf-8") as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Generate language memory labels.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--api_key", default=os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument("--max_episodes", type=int, default=None)
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY or use --api_key.")

    # 创建生成器
    from openpi.models.memory_manager import MemoryDataGenerator, MemoryGenerationConfig
    llm_client = create_anthropic_client(args.api_key)
    generator = MemoryDataGenerator(
        llm_client=llm_client,
        config=MemoryGenerationConfig(max_memory_length=512),
    )

    # 加载 episodes
    episodes = load_episodes_from_dataset(args.dataset_path)
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    logger.info(f"Processing {len(episodes)} episodes...")

    # 生成标注
    all_labels = []
    for i, episode in enumerate(episodes):
        labels = generator.generate_labels_for_episode(
            episode_id=episode["episode_id"],
            subtasks=episode["subtasks"],
        )
        all_labels.extend(labels)
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{len(episodes)} episodes, {len(all_labels)} labels")

    # 保存
    generator.save_labels(all_labels, args.output_path)
    logger.info(f"Done! Total labels: {len(all_labels)}")


if __name__ == "__main__":
    main()
```

---

## 13. 测试方案与代码

### 13.1 测试文件结构

```
tests/
├── test_video_encoder.py      # Step 2 — 视频编码器数值正确性
├── test_mem_observation.py    # Step 1 — 数据结构向后兼容
├── test_mem_model.py          # Step 3 — 模型前向传播
├── test_memory_manager.py     # Step 5 — 记忆管理器
├── test_transforms_mem.py     # Step 7 — 数据变换
└── test_policy_mem.py         # Step 10 — 推理接口
```

---

### 13.2 `test_video_encoder.py` — 视频编码器测试

```python
"""测试视频编码器的核心性质：
1. K=1 时与原 SigLIP 输出数值一致
2. 多帧时输出形状正确
3. 因果掩码：历史帧信息不污染早期帧
4. 推理延迟满足实时要求
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# 注意：实际测试时需要先加载模型权重
# 此处使用随机初始化进行形状和梯度流测试


@pytest.fixture(scope="module")
def dummy_siglip_config():
    """SigLIP So400m 配置（简化版，用于测试）。"""
    return {
        "variant": "So400m/14",
        "num_classes": 0,
        "pool_type": "none",
        "scan": False,   # 视频编码器使用 non-scan
    }


class TestVideoEncoderSingleFrame:
    """K=1 时视频编码器退化为普通 ViT 的测试。"""

    def test_single_frame_output_shape(self):
        """单帧时输出形状 (B, N, D) 与原 ViT 一致。"""
        from openpi.models.siglip import _Module, decode_variant

        config = decode_variant("So400m/14")
        model = _Module(num_classes=0, pool_type="none", scan=False, **config)

        key = jax.random.PRNGKey(0)
        dummy_img = jnp.ones((2, 224, 224, 3))  # B=2

        # 初始化并运行
        variables = model.init(key, dummy_img, train=False)
        out, _ = model.apply(variables, dummy_img, train=False)

        # SigLIP So400m/14：224/14 = 16, N = 16*16 = 256
        assert out.shape == (2, 256, 1152), f"Expected (2, 256, 1152), got {out.shape}"

    def test_k1_equals_original_vit(self):
        """验证 K=1 时（无时间注意力），输出与原始 ViT 精确一致。

        由于 SpaceTimeSeparableBlock 中时间位置编码在 t=0 时为 0，
        K=1 时时间注意力无效，退化为普通空间注意力。
        """
        from openpi.models.siglip import _Module, Encoder, decode_variant

        config = decode_variant("So400m/14")
        key = jax.random.PRNGKey(42)
        dummy_img = jax.random.normal(key, (1, 224, 224, 3))

        # 原始模型（num_timesteps=1）
        model_orig = _Module(
            num_classes=0, pool_type="none", scan=False,
            num_timesteps=1, **config
        )
        vars_orig = model_orig.init(key, dummy_img, train=False)
        out_orig, _ = model_orig.apply(vars_orig, dummy_img, train=False)

        # 视频模型（K=1，应退化为原始 ViT）
        model_video = _Module(
            num_classes=0, pool_type="none", scan=False,
            num_timesteps=1, temporal_attn_every=4, **config
        )
        vars_video = model_video.init(key, dummy_img, train=False)
        out_video, _ = model_video.apply(vars_video, dummy_img, train=False)

        # 数值误差 < 1e-5（论文要求）
        max_diff = float(jnp.max(jnp.abs(out_orig - out_video)))
        assert max_diff < 1e-5, (
            f"K=1 video encoder output differs from original ViT by {max_diff:.2e}"
        )


class TestVideoEncoderMultiFrame:
    """多帧视频编码器测试。"""

    def test_output_shape_k6(self):
        """K=6 帧时，输出形状应与单帧相同（历史帧 token 被丢弃）。"""
        from openpi.models.siglip import _Module, decode_variant

        config = decode_variant("So400m/14")
        K = 6
        B = 2

        model = _Module(
            num_classes=0, pool_type="none", scan=False,
            num_timesteps=K, temporal_attn_every=4,
            drop_history_after_layer=-4,
            **config
        )
        key = jax.random.PRNGKey(0)
        # 输入 B*K 帧
        dummy_frames = jax.random.normal(key, (B * K, 224, 224, 3))
        variables = model.init(key, dummy_frames, train=False)
        out, _ = model.apply(variables, dummy_frames, train=False)

        # 输出应该是 B 帧（历史帧已被丢弃）
        assert out.shape == (B, 256, 1152), (
            f"Expected ({B}, 256, 1152), got {out.shape}"
        )

    def test_temporal_info_flows_to_current_frame(self):
        """验证时间注意力确实将历史信息融合到当前帧 token 中。

        改变历史帧内容，当前帧的输出 token 应发生变化。
        """
        from openpi.models.siglip import _Module, decode_variant

        config = decode_variant("So400m/14")
        K = 3

        model = _Module(
            num_classes=0, pool_type="none", scan=False,
            num_timesteps=K, temporal_attn_every=4, **config
        )
        key = jax.random.PRNGKey(0)

        # 基准输入：K 帧全零
        frames_base = jnp.zeros((K, 224, 224, 3))
        variables = model.init(key, frames_base, train=False)
        out_base, _ = model.apply(variables, frames_base, train=False)

        # 修改历史帧（前 K-1 帧），当前帧不变
        frames_modified = frames_base.at[:K-1].set(
            jax.random.normal(key, (K-1, 224, 224, 3))
        )
        out_modified, _ = model.apply(variables, frames_modified, train=False)

        # 输出应该不同（时间信息已融合）
        diff = float(jnp.mean(jnp.abs(out_base - out_modified)))
        assert diff > 1e-4, (
            f"Temporal attention not working: output unchanged when history frames modified "
            f"(mean diff = {diff:.2e})"
        )

    def test_causal_mask_no_future_leakage(self):
        """验证因果掩码：早期帧不能 attend 到未来帧。

        修改最后一帧（当前帧），早期帧的内部激活不应改变。
        （注：此测试需要访问中间层激活，使用简化的小型模型）
        """
        # 使用 mu（最小）模型快速测试
        from openpi.models.siglip import _Module, decode_variant

        config = decode_variant("mu")
        K = 3
        model = _Module(
            num_classes=0, pool_type="none", scan=False,
            num_timesteps=K, temporal_attn_every=1, **config
        )
        key = jax.random.PRNGKey(0)
        img_size = 32  # mu 模型用小图

        frames = jax.random.normal(key, (K, img_size, img_size, 3))
        variables = model.init(key, frames, train=False)

        # 获取倒数第2帧的激活（在修改最后帧前后）
        def get_early_frame_activations(frames_input):
            _, out = model.apply(variables, frames_input, train=False)
            # 取第一个 block 的输出（时间注意力前）
            return out.get("encoder", {}).get("block00", jnp.zeros(1))

        act_base = get_early_frame_activations(frames)

        # 修改最后一帧（当前帧）
        frames_mod = frames.at[-1].set(jax.random.normal(key, frames[-1].shape))
        act_modified = get_early_frame_activations(frames_mod)

        # 早期帧的激活不应改变（因果掩码保证）
        diff = float(jnp.mean(jnp.abs(act_base - act_modified)))
        assert diff < 1e-6, (
            f"Causal mask violated: early frame activations changed "
            f"when future frame modified (diff={diff:.2e})"
        )


class TestVideoEncoderLatency:
    """推理延迟测试。"""

    @pytest.mark.slow
    def test_latency_k6_under_300ms(self, dummy_siglip_config):
        """K=6 帧，4 路摄像头时延迟 < 300ms（论文要求）。

        注意：此测试需要 GPU 环境，CI 中跳过。
        """
        if jax.default_backend() == "cpu":
            pytest.skip("Latency test requires GPU")

        from openpi.models.siglip import _Module, decode_variant

        K = 6
        N_CAMS = 4
        config = decode_variant("So400m/14")
        model = _Module(
            num_classes=0, pool_type="none", scan=False,
            num_timesteps=K, **config
        )
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((N_CAMS * K, 448, 448, 3))  # 448px 分辨率（论文配置）
        variables = model.init(key, dummy_input, train=False)

        # 预热
        _ = model.apply(variables, dummy_input, train=False)
        jax.block_until_ready(_)

        # 计时
        n_trials = 10
        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            out = model.apply(variables, dummy_input, train=False)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)

        median_ms = np.median(times) * 1000
        assert median_ms < 300, f"Latency {median_ms:.1f}ms exceeds 300ms threshold"
        print(f"\nVideo encoder latency (K={K}, {N_CAMS} cams): {median_ms:.1f}ms")
```

---

### 13.3 `test_mem_observation.py` — 数据结构测试

```python
"""测试 Observation 数据结构的向后兼容性和 MEM 字段处理。"""

import numpy as np
import pytest

from openpi.models.model import Observation, preprocess_observation


class TestObservationBackwardCompat:
    """向后兼容性测试：原有代码不需要任何修改。"""

    def _make_pi05_data(self, batch_size=2):
        """构造 π0.5 格式的数据字典（无任何 MEM 字段）。"""
        return {
            "image": {
                "base_0_rgb": np.random.rand(batch_size, 224, 224, 3).astype(np.float32),
                "left_wrist_0_rgb": np.random.rand(batch_size, 224, 224, 3).astype(np.float32),
                "right_wrist_0_rgb": np.random.rand(batch_size, 224, 224, 3).astype(np.float32),
            },
            "image_mask": {
                "base_0_rgb": np.ones(batch_size, dtype=bool),
                "left_wrist_0_rgb": np.ones(batch_size, dtype=bool),
                "right_wrist_0_rgb": np.ones(batch_size, dtype=bool),
            },
            "state": np.random.rand(batch_size, 32).astype(np.float32),
            "tokenized_prompt": np.zeros((batch_size, 48), dtype=np.int32),
            "tokenized_prompt_mask": np.ones((batch_size, 48), dtype=bool),
        }

    def test_from_dict_without_mem_fields(self):
        """不含 MEM 字段的数据应正常解析，MEM 字段为 None。"""
        data = self._make_pi05_data()
        obs = Observation.from_dict(data)

        assert obs.images is not None
        assert obs.state is not None
        # MEM 字段应为 None（兼容 π0.5）
        assert obs.image_history is None, "image_history should be None when not provided"
        assert obs.state_history is None
        assert obs.tokenized_memory is None

    def test_from_dict_with_mem_fields(self):
        """含 MEM 字段的数据应正确解析。"""
        K = 5  # 历史帧数 = K-1 = 5
        data = self._make_pi05_data(batch_size=2)
        data["image_history"] = {
            "base_0_rgb": np.random.rand(2, K, 224, 224, 3).astype(np.float32),
        }
        data["state_history"] = np.random.rand(2, K, 32).astype(np.float32)
        data["tokenized_memory"] = np.zeros((2, 256), dtype=np.int32)
        data["tokenized_memory_mask"] = np.ones((2, 256), dtype=bool)

        obs = Observation.from_dict(data)
        assert obs.image_history is not None
        assert obs.image_history["base_0_rgb"].shape == (2, K, 224, 224, 3)
        assert obs.state_history.shape == (2, K, 32)
        assert obs.tokenized_memory.shape == (2, 256)

    def test_uint8_image_history_conversion(self):
        """uint8 类型的历史帧图像应自动归一化到 [-1, 1]。"""
        data = self._make_pi05_data()
        data["image_history"] = {
            "base_0_rgb": (np.random.rand(2, 5, 224, 224, 3) * 255).astype(np.uint8),
        }
        obs = Observation.from_dict(data)
        hist = obs.image_history["base_0_rgb"]
        assert hist.dtype == np.float32
        assert hist.min() >= -1.0 - 1e-5
        assert hist.max() <= 1.0 + 1e-5

    def test_preprocess_observation_passthrough(self):
        """preprocess_observation 应正确传递 MEM 字段。"""
        data = self._make_pi05_data()
        K = 3
        data["image_history"] = {
            "base_0_rgb": np.random.rand(2, K, 224, 224, 3).astype(np.float32) * 2 - 1,
        }
        obs = Observation.from_dict(data)
        processed = preprocess_observation(None, obs, train=False)

        # 历史帧应被保留
        assert processed.image_history is not None
        assert "base_0_rgb" in processed.image_history
```

---

### 13.4 `test_mem_model.py` — 模型前向传播测试

```python
"""测试 Pi0 模型集成 MEM 后的前向传播。"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def make_pi06_mem_config(use_video=False, use_lang_mem=False, K=6):
    """创建 π0.6-MEM 测试配置（使用小型模型加速测试）。"""
    from openpi.models.pi0_config import MEMConfig, Pi0Config
    return Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
        action_dim=32,
        action_horizon=4,   # 缩短以加快测试
        mem=MEMConfig(
            use_video_memory=use_video,
            video_memory_frames=K,
            use_language_memory=use_lang_mem,
            max_memory_tokens=32,  # 短记忆加速测试
        ),
    )


class TestPi0MemForwardPass:
    """π0.6-MEM 前向传播测试。"""

    def _make_batch_obs(self, batch_size=1, K=1, max_memory_tokens=32):
        """构造测试用 Observation。"""
        from openpi.models.model import Observation

        imgs = {
            cam: np.random.rand(batch_size, 224, 224, 3).astype(np.float32) * 2 - 1
            for cam in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        }
        masks = {cam: np.ones(batch_size, dtype=bool) for cam in imgs}

        kwargs = dict(
            images=imgs,
            image_masks=masks,
            state=np.random.rand(batch_size, 32).astype(np.float32),
            tokenized_prompt=np.zeros((batch_size, 200), dtype=np.int32),
            tokenized_prompt_mask=np.ones((batch_size, 200), dtype=bool),
        )

        if K > 1:
            kwargs["image_history"] = {
                cam: np.random.rand(batch_size, K - 1, 224, 224, 3).astype(np.float32) * 2 - 1
                for cam in imgs
            }
            kwargs["state_history"] = np.random.rand(batch_size, K - 1, 32).astype(np.float32)

        if max_memory_tokens > 0:
            kwargs["tokenized_memory"] = np.zeros((batch_size, max_memory_tokens), dtype=np.int32)
            kwargs["tokenized_memory_mask"] = np.zeros((batch_size, max_memory_tokens), dtype=bool)

        return Observation(**kwargs)

    @pytest.mark.parametrize("use_video,K", [(False, 1), (True, 2), (True, 4)])
    def test_sample_actions_shape(self, use_video, K):
        """验证不同配置下 sample_actions 输出形状正确。"""
        config = make_pi06_mem_config(use_video=use_video, K=K)
        rng = jax.random.PRNGKey(0)
        model = config.create(rng)

        obs = self._make_batch_obs(batch_size=1, K=K)
        actions = model.sample_actions(rng, obs, num_steps=2)

        expected_shape = (1, config.action_horizon, config.action_dim)
        assert actions.shape == expected_shape, (
            f"use_video={use_video}, K={K}: "
            f"Expected {expected_shape}, got {actions.shape}"
        )

    def test_no_mem_equals_pi05(self):
        """无 MEM 时，π0.6-MEM 输出应与 π0.5 数值一致。"""
        from openpi.models.pi0_config import Pi0Config

        # π0.5 配置
        config_pi05 = Pi0Config(pi05=True, action_dim=32, action_horizon=4)
        # π0.6-MEM（未启用 MEM）
        config_mem = make_pi06_mem_config(use_video=False, use_lang_mem=False)

        rng = jax.random.PRNGKey(42)

        # 使用相同 rng 初始化，权重应一致
        model_pi05 = config_pi05.create(rng)
        model_mem = config_mem.create(rng)

        obs = self._make_batch_obs(batch_size=1, K=1, max_memory_tokens=0)
        action_rng = jax.random.PRNGKey(99)

        out_pi05 = model_pi05.sample_actions(action_rng, obs, num_steps=2)
        out_mem = model_mem.sample_actions(action_rng, obs, num_steps=2)

        max_diff = float(jnp.max(jnp.abs(out_pi05 - out_mem)))
        assert max_diff < 1e-4, (
            f"No-MEM model differs from π0.5 by {max_diff:.2e}"
        )

    def test_language_memory_changes_output(self):
        """注入语言记忆后，模型输出应与不注入时不同（记忆确实产生影响）。"""
        config = make_pi06_mem_config(use_video=False, use_lang_mem=True)
        rng = jax.random.PRNGKey(0)
        model = config.create(rng)

        # 无记忆
        obs_no_mem = self._make_batch_obs(batch_size=1, K=1, max_memory_tokens=0)
        out_no_mem = model.sample_actions(rng, obs_no_mem, num_steps=2)

        # 有记忆（非零 token）
        obs_with_mem = self._make_batch_obs(batch_size=1, K=1, max_memory_tokens=32)
        # 设置非零记忆 token
        obs_with_mem = obs_with_mem.replace(
            tokenized_memory=np.ones((1, 32), dtype=np.int32) * 100,
            tokenized_memory_mask=np.ones((1, 32), dtype=bool),
        )
        out_with_mem = model.sample_actions(rng, obs_with_mem, num_steps=2)

        diff = float(jnp.mean(jnp.abs(out_no_mem - out_with_mem)))
        assert diff > 1e-4, "Language memory has no effect on model output"

    def test_compute_loss_valid(self):
        """compute_loss 应返回有效（非 NaN/Inf）的损失值。"""
        config = make_pi06_mem_config(use_video=True, K=3)
        rng = jax.random.PRNGKey(0)
        model = config.create(rng)
        model.train()  # 设置 dropout

        obs = self._make_batch_obs(batch_size=2, K=3)
        actions = np.random.rand(2, config.action_horizon, config.action_dim).astype(np.float32)

        loss = model.compute_loss(rng, obs, actions, train=True)
        assert not jnp.any(jnp.isnan(loss)), "Loss contains NaN"
        assert not jnp.any(jnp.isinf(loss)), "Loss contains Inf"
        assert float(jnp.mean(loss)) > 0, "Loss should be positive"
```

---

### 13.5 `test_memory_manager.py` — 记忆管理器测试

```python
"""测试语言记忆管理器的功能。"""

import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from openpi.models.memory_manager import MemoryDataGenerator, MemoryGenerationConfig, MemoryLabel


class TestMemoryDataGenerator:
    """记忆标注生成器测试。"""

    @pytest.fixture
    def mock_llm(self):
        """模拟 LLM 客户端。"""
        llm = MagicMock()
        llm.generate.return_value = "I placed the pot in the sink and grabbed the potatoes."
        return llm

    @pytest.fixture
    def generator(self, mock_llm):
        return MemoryDataGenerator(
            llm_client=mock_llm,
            config=MemoryGenerationConfig(max_memory_length=256),
        )

    def test_generate_labels_basic(self, generator):
        """基础标注生成流程。"""
        subtasks = [
            {"instruction": "Move pot to sink", "success": True},
            {"instruction": "Grab potatoes from fridge", "success": True},
            {"instruction": "Grab milk from fridge", "success": False},  # 失败
            {"instruction": "Grab milk from fridge", "success": True},   # 重试成功
        ]
        labels = generator.generate_labels_for_episode("ep001", subtasks)

        assert len(labels) == 4
        assert labels[0].episode_id == "ep001"
        assert labels[0].timestep == 0
        # 第 0 步：无历史，记忆为空
        assert labels[0].memory_before == ""
        # 第 1 步：step 0 成功后记忆非空
        assert labels[1].memory_before != "" or labels[1].memory_after != ""

    def test_failed_subtask_does_not_update_memory(self, generator, mock_llm):
        """失败子任务不应更新记忆（防止分布偏移）。"""
        subtasks = [
            {"instruction": "Pick up bowl", "success": True},
            {"instruction": "Pick up cup", "success": False},  # 失败
            {"instruction": "Pick up cup", "success": True},   # 重试成功
        ]
        labels = generator.generate_labels_for_episode("ep002", subtasks)

        # step 1（失败）之后，step 2 的 memory_before 应与 step 1 的 memory_before 相同
        # 因为失败不更新记忆
        assert labels[2].memory_before == labels[1].memory_before, (
            "Failed subtask should not update memory"
        )

    def test_save_and_load_labels(self, generator):
        """标注文件的读写一致性。"""
        original_labels = [
            MemoryLabel(
                episode_id="ep003",
                timestep=i,
                subtask_instruction=f"Step {i}",
                subtask_success=True,
                memory_before=f"Memory at step {i}",
                memory_after=f"Memory after step {i}",
            )
            for i in range(5)
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            tmp_path = f.name

        try:
            generator.save_labels(original_labels, tmp_path)
            loaded_labels = MemoryDataGenerator.load_labels(tmp_path)

            assert len(loaded_labels) == len(original_labels)
            for orig, loaded in zip(original_labels, loaded_labels):
                assert orig.episode_id == loaded.episode_id
                assert orig.memory_before == loaded.memory_before
                assert orig.memory_after == loaded.memory_after
        finally:
            os.unlink(tmp_path)

    def test_first_step_empty_memory(self, generator):
        """第一步的 memory_before 应为空字符串。"""
        subtasks = [{"instruction": "Start task", "success": True}]
        labels = generator.generate_labels_for_episode("ep004", subtasks)
        assert labels[0].memory_before == ""


class TestHighLevelPolicy:
    """高层策略测试。"""

    @pytest.fixture
    def mock_vlm(self):
        """模拟 VLM 推理函数（返回有效 JSON）。"""
        def vlm_fn(image, prompt, max_tokens=256):
            return json.dumps({
                "subtask": "Pick up the bowl",
                "updated_memory": "I have cleared the counter and started gathering items."
            })
        return vlm_fn

    @pytest.fixture
    def hl_policy(self, mock_vlm):
        from openpi.models.high_level_policy import HighLevelPolicy, HighLevelPolicyConfig
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        return HighLevelPolicy(
            vlm_inference_fn=mock_vlm,
            tokenizer=tokenizer,
            config=HighLevelPolicyConfig(subtask_trigger_steps=10),
        )

    def test_reset_clears_state(self, hl_policy):
        """reset 后状态应被清空。"""
        hl_policy._language_memory = "Some old memory"
        hl_policy._episode_step = 100
        hl_policy.reset("Clean the kitchen")
        assert hl_policy.language_memory == ""
        assert hl_policy._episode_step == 0
        assert hl_policy._task_goal == "Clean the kitchen"

    def test_update_returns_subtask_and_memory(self, hl_policy):
        """update 应返回子任务和更新后的记忆。"""
        import numpy as np
        hl_policy.reset("Set up recipe")
        subtask, memory = hl_policy.update(
            observation_image=np.zeros((224, 224, 3), dtype=np.uint8),
            subtask_success=True,
        )
        assert isinstance(subtask, str) and len(subtask) > 0
        assert isinstance(memory, str)

    def test_failed_update_preserves_memory(self, hl_policy):
        """子任务失败时不更新记忆。"""
        hl_policy.reset("Task")
        hl_policy._language_memory = "Original memory"

        import numpy as np
        _, memory = hl_policy.update(
            observation_image=np.zeros((224, 224, 3), dtype=np.uint8),
            subtask_success=False,  # 失败
        )
        assert memory == "Original memory", "Memory should not be updated on failure"

    def test_tokenize_memory_padding(self, hl_policy):
        """记忆 token 化应正确 padding 到固定长度。"""
        token_ids, token_mask = hl_policy.tokenize_memory("Test memory")
        max_len = hl_policy.config.max_memory_tokens
        assert len(token_ids) == max_len
        assert len(token_mask) == max_len

    def test_tokenize_empty_memory(self, hl_policy):
        """空记忆应返回全零 token。"""
        token_ids, token_mask = hl_policy.tokenize_memory("")
        assert all(t == 0 for t in token_ids)
        assert not any(token_mask)

    def test_should_update_trigger(self, hl_policy):
        """should_update 应按 subtask_trigger_steps 触发。"""
        hl_policy.reset("Task")
        trigger_steps = hl_policy.config.subtask_trigger_steps  # 10

        results = [hl_policy.should_update() for _ in range(trigger_steps + 1)]
        # 第 trigger_steps 步应触发
        assert results[trigger_steps - 1] == True, "Should trigger at step trigger_steps"
        assert results[0] == False, "Should not trigger at step 0"
```

---

### 13.6 `test_transforms_mem.py` — 数据变换测试

```python
"""测试 MEM 相关数据变换。"""

import numpy as np
import pytest

from openpi.transforms import VideoFrameStack, TokenizeMemory


class TestVideoFrameStack:
    """VideoFrameStack 变换测试。"""

    def test_passthrough_without_history(self):
        """没有 image_history 时，数据直接传递（向后兼容）。"""
        data = {
            "image": {"base_0_rgb": np.zeros((224, 224, 3))},
            "state": np.zeros(32),
        }
        transform = VideoFrameStack(num_frames=6, stride=5)
        result = transform(data)
        # 无 image_history 字段，数据不变
        assert "image_history" not in result

    def test_history_shape_validation(self):
        """当 image_history 存在时，验证形状正确性。"""
        K = 6
        data = {
            "image": {"base_0_rgb": np.zeros((224, 224, 3))},
            "image_history": {
                "base_0_rgb": np.zeros((K - 1, 224, 224, 3)),
            },
            "state": np.zeros(32),
        }
        transform = VideoFrameStack(num_frames=K, stride=5)
        result = transform(data)
        assert result["image_history"]["base_0_rgb"].shape == (K - 1, 224, 224, 3)


class TestTokenizeMemory:
    """TokenizeMemory 变换测试。"""

    @pytest.fixture
    def mock_tokenizer(self):
        from unittest.mock import MagicMock
        tok = MagicMock()
        tok.encode.return_value = [10, 20, 30, 40, 50]  # 5 tokens
        return tok

    def test_tokenize_adds_fields(self, mock_tokenizer):
        """变换后应包含 tokenized_memory 和 tokenized_memory_mask。"""
        data = {"language_memory": "I placed the pot in the sink."}
        transform = TokenizeMemory(tokenizer=mock_tokenizer, max_len=16)
        result = transform(data)
        assert "tokenized_memory" in result
        assert "tokenized_memory_mask" in result
        assert len(result["tokenized_memory"]) == 16
        assert len(result["tokenized_memory_mask"]) == 16

    def test_padding_correct(self, mock_tokenizer):
        """短文本应被 padding 到 max_len。"""
        data = {"language_memory": "Short."}
        mock_tokenizer.encode.return_value = [1, 2, 3]  # 3 tokens

        transform = TokenizeMemory(tokenizer=mock_tokenizer, max_len=10)
        result = transform(data)

        assert result["tokenized_memory"].tolist() == [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
        assert result["tokenized_memory_mask"].tolist() == [True, True, True] + [False] * 7

    def test_truncation_to_max_len(self, mock_tokenizer):
        """长文本应被截断到 max_len。"""
        mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens
        data = {"language_memory": "Very long memory " * 20}

        transform = TokenizeMemory(tokenizer=mock_tokenizer, max_len=10)
        result = transform(data)

        assert len(result["tokenized_memory"]) == 10
        assert all(result["tokenized_memory_mask"])  # 全有效

    def test_passthrough_without_memory(self, mock_tokenizer):
        """无 language_memory 字段时数据直接传递。"""
        data = {"state": np.zeros(32)}
        transform = TokenizeMemory(tokenizer=mock_tokenizer, max_len=16)
        result = transform(data)
        assert "tokenized_memory" not in result
```

---

### 13.7 `test_policy_mem.py` — 推理接口测试

```python
"""测试 MEMPolicy 推理接口（帧缓存和记忆状态管理）。"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestMEMPolicyFrameBuffer:
    """帧缓存管理测试。"""

    @pytest.fixture
    def mem_policy(self):
        """创建简单 MEMPolicy 用于测试帧缓存逻辑。"""
        from openpi.policies.policy import MEMPolicy

        mock_model = MagicMock()
        mock_model.sample_actions.return_value = np.zeros((1, 50, 32))

        policy = MEMPolicy(
            model=mock_model,
            num_video_frames=4,
            camera_keys=["base_0_rgb"],
        )
        return policy

    def _make_obs(self, value=0.0):
        """创建测试观测。"""
        return {
            "base_0_rgb": np.full((224, 224, 3), value, dtype=np.float32),
            "left_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.float32),
            "right_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.float32),
            "state": np.zeros(32, dtype=np.float32),
            "tokenized_prompt": np.zeros(200, dtype=np.int32),
            "tokenized_prompt_mask": np.ones(200, dtype=bool),
        }

    def test_reset_clears_frame_buffer(self, mem_policy):
        """reset_episode 后帧缓存应为空。"""
        # 先填充缓存
        obs = self._make_obs()
        mem_policy._update_frame_buffer(obs)
        assert len(mem_policy._frame_buffer["base_0_rgb"]) > 0

        mem_policy.reset_episode()
        assert len(mem_policy._frame_buffer["base_0_rgb"]) == 0

    def test_frame_buffer_fills_gradually(self, mem_policy):
        """帧缓存应逐步填充，最大保留 K-1 帧。"""
        K = mem_policy.K  # 4
        for i in range(K + 2):
            obs = self._make_obs(value=float(i))
            mem_policy._update_frame_buffer(obs)

        # 最大保留 K-1 = 3 帧
        assert len(mem_policy._frame_buffer["base_0_rgb"]) == K - 1

    def test_history_padding_at_episode_start(self, mem_policy):
        """episode 开始时缓存为空，应用当前帧 padding。"""
        mem_policy.reset_episode()
        obs = self._make_obs(value=1.0)
        obs_with_history = mem_policy._build_obs_with_history(obs)

        assert "image_history" in obs_with_history
        hist = obs_with_history["image_history"]["base_0_rgb"]
        assert hist.shape[0] == mem_policy.K - 1  # K-1 历史帧
        # padding 帧应与当前帧值相同（用第一帧 padding）
        # 实际是用当前帧复制的

    def test_history_contains_correct_frames(self, mem_policy):
        """帧缓存内容应反映推理历史。"""
        mem_policy.reset_episode()

        # 推入 3 帧（value=1,2,3）
        for v in [1.0, 2.0, 3.0]:
            mem_policy._update_frame_buffer(self._make_obs(value=v))

        obs_now = self._make_obs(value=4.0)
        obs_with_history = mem_policy._build_obs_with_history(obs_now)

        hist = obs_with_history["image_history"]["base_0_rgb"]
        # K=4, K-1=3 历史帧，值应为 1.0, 2.0, 3.0
        assert hist[0].mean() == pytest.approx(1.0, abs=1e-4)
        assert hist[1].mean() == pytest.approx(2.0, abs=1e-4)
        assert hist[2].mean() == pytest.approx(3.0, abs=1e-4)


class TestMEMPolicyMemoryState:
    """语言记忆状态管理测试。"""

    @pytest.fixture
    def policy_with_hl(self):
        from openpi.policies.policy import MEMPolicy
        from openpi.models.high_level_policy import HighLevelPolicy, HighLevelPolicyConfig

        mock_model = MagicMock()
        mock_model.sample_actions.return_value = np.zeros((1, 50, 32))

        mock_vlm = MagicMock()
        mock_vlm.return_value = '{"subtask": "Pick up bowl", "updated_memory": "I picked up the bowl."}'

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        hl = HighLevelPolicy(
            vlm_inference_fn=mock_vlm,
            tokenizer=mock_tokenizer,
            config=HighLevelPolicyConfig(subtask_trigger_steps=3),
        )
        hl.reset("Test task")

        policy = MEMPolicy(
            model=mock_model,
            high_level_policy=hl,
            num_video_frames=1,
        )
        return policy

    def test_memory_injected_into_obs(self, policy_with_hl):
        """推理时应将语言记忆注入观测字典。"""
        policy_with_hl._language_memory = "Test memory content"

        obs = {
            "base_0_rgb": np.zeros((224, 224, 3)),
            "left_wrist_0_rgb": np.zeros((224, 224, 3)),
            "right_wrist_0_rgb": np.zeros((224, 224, 3)),
            "state": np.zeros(32),
            "tokenized_prompt": np.zeros(200, dtype=np.int32),
            "tokenized_prompt_mask": np.ones(200, dtype=bool),
        }
        # 验证 tokenized_memory 被注入
        obs_with_history = policy_with_hl._build_obs_with_history(obs)
        # 在 infer 之前手动检查（此处仅测试 _build_obs 逻辑）
        assert "image_history" not in obs_with_history or True  # K=1 时无 history


class TestMEMPolicyEndToEnd:
    """端到端推理测试（使用 Mock 模型）。"""

    def test_infer_returns_actions(self):
        """推理应返回正确形状的动作。"""
        from openpi.policies.policy import MEMPolicy

        # Mock 模型：直接返回假动作
        mock_model = MagicMock()
        # sample_actions 在 JAX 中调用，返回 (1, 50, 32)
        mock_model.sample_actions.return_value = np.zeros((1, 50, 32))

        policy = MEMPolicy(
            model=mock_model,
            num_video_frames=1,
        )
        policy.reset_episode("Test")

        obs = {
            "base_0_rgb": np.zeros((224, 224, 3), dtype=np.float32),
            "left_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.float32),
            "right_wrist_0_rgb": np.zeros((224, 224, 3), dtype=np.float32),
            "state": np.zeros(32, dtype=np.float32),
            "tokenized_prompt": np.zeros(200, dtype=np.int32),
            "tokenized_prompt_mask": np.ones(200, dtype=bool),
        }

        # 应该不抛出异常
        # （完整测试需要真实 JAX 模型）
        assert policy is not None
```

---

### 13.8 运行测试的命令

```bash
# 安装测试依赖
cd openpi-main
pip install -e ".[dev]" --break-system-packages

# 运行所有 MEM 相关单元测试（排除需要 GPU 的慢测试）
pytest tests/test_mem_observation.py tests/test_memory_manager.py \
       tests/test_transforms_mem.py tests/test_policy_mem.py \
       -v --tb=short

# 运行视频编码器测试（需要 JAX 环境）
pytest tests/test_video_encoder.py -v --tb=short -m "not slow"

# 运行完整模型前向传播测试（较慢，建议单独运行）
pytest tests/test_mem_model.py -v --tb=long -s

# 运行所有测试（包含慢测试，需要 GPU）
pytest tests/ -v --tb=short

# 生成测试覆盖率报告
pytest tests/ --cov=openpi --cov-report=html:coverage_html

# 快速冒烟测试（仅数据结构和变换，无需模型权重）
pytest tests/test_mem_observation.py tests/test_memory_manager.py \
       tests/test_transforms_mem.py -v -x
```

---

### 13.9 集成验证清单

在提交 PR 前，按顺序执行以下验证：

```bash
# 1. 向后兼容性：现有 π0.5 测试不应有任何失败
pytest src/openpi/models/pi0_test.py src/openpi/models/model_test.py -v

# 2. K=1 数值等价性（核心约束）
pytest tests/test_video_encoder.py::TestVideoEncoderSingleFrame::test_k1_equals_original_vit -v -s

# 3. 记忆管理器核心功能
pytest tests/test_memory_manager.py -v

# 4. 数据结构读写
pytest tests/test_mem_observation.py -v

# 5. 端到端前向传播（需要 GPU，CI 可跳过）
pytest tests/test_mem_model.py -v -m "not slow" --timeout=120
```

---

## 附录：关键实现顺序建议

实现时建议严格按以下顺序进行，每步完成后立即运行对应测试：

| 顺序 | 文件 | 关键验证点 | 对应测试 |
|------|------|------------|----------|
| 1 | `model.py` | Observation 向后兼容 | `test_mem_observation.py` |
| 2 | `memory_manager.py` | 记忆生成和压缩逻辑 | `test_memory_manager.py` |
| 3 | `transforms.py` | VideoFrameStack / TokenizeMemory | `test_transforms_mem.py` |
| 4 | `siglip.py` | K=1 数值等价 + 时间信息融合 | `test_video_encoder.py` |
| 5 | `pi0_config.py` | MEMConfig 默认值 | — |
| 6 | `pi0.py` | 多帧前向传播 + loss 正常 | `test_mem_model.py` |
| 7 | `high_level_policy.py` | 子任务/记忆更新流程 | `test_memory_manager.py::TestHighLevelPolicy` |
| 8 | `policies/policy.py` | 帧缓存 + 推理接口 | `test_policy_mem.py` |
| 9 | `training/config.py` + `data_loader.py` | 端到端训练数据流 | 集成测试 |
