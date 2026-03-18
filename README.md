# π0.6-MEM: Multi-Scale Embodied Memory for Vision-Language-Action Models

> 基于 openpi (π0.5) 代码库实现的 Multi-Scale Embodied Memory (MEM) 架构  
> 参考论文：[MEM: Multi-Scale Embodied Memory for Vision Language Action Models](https://pi.website/research/memory)

---

## 项目概述

π0.6-MEM 是 Physical Intelligence π0.5 模型的扩展版本，集成了 Multi-Scale Embodied Memory (MEM) 架构。该架构通过多尺度记忆机制增强了 VLA (Vision-Language-Action) 模型在长时序机器人任务中的性能。

### 核心特性

- **短期视觉记忆**：通过时空分离注意力机制融合历史帧信息
- **长期语言记忆**：使用 LLM 生成和管理语义记忆
- **本体感觉历史**：连续状态嵌入增强动作预测
- **向后兼容**：MEM 禁用时行为与 π0.5 完全一致

---

## 架构设计

### 记忆层级

```
┌─────────────────────────────────────────────────────────────┐
│                    Long-term Language Memory                │
│                 (LLM-generated semantic memory)             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Short-term Visual Memory (K frames)            │
│         SpaceTimeSeparableBlock + Temporal Attention        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Current Observation                     │
│              (image tokens + state tokens)                  │
└─────────────────────────────────────────────────────────────┘
```

### 关键设计

| 特性 | 实现 |
|------|------|
| 时空分离注意力 | 每 4 层使用一次 SpaceTimeSeparableBlock |
| 因果时间注意力 | 当前帧只能 attend 到历史帧 |
| 历史帧 token 丢弃 | 在上层丢弃，减少计算量 |
| 语言记忆压缩 | LLM 生成压缩后的记忆摘要 |

---

## 目录结构

```
src/openpi/
├── models/
│   ├── siglip.py              ← SpaceTimeSeparableBlock + 视频编码器
│   ├── model.py               ← Observation 新增历史帧字段
│   ├── pi0.py                 ← MEM 前向传播逻辑
│   ├── pi0_config.py          ← MEMConfig 配置类
│   ├── high_level_policy.py   ← 高层策略（子任务 + 记忆更新）
│   └── memory_manager.py       ← 语言记忆管理
├── policies/
│   └── policy.py              ← MEMPolicy 推理接口
├── training/
│   └── data_loader.py         ← MEMLeRobotDataset
├── transforms.py              ← VideoFrameStack + TokenizeMemory
└── scripts/
    └── gen_memory_labels.py   ← 记忆标注生成脚本
tests/
├── test_video_encoder.py
├── test_mem_model.py
├── test_memory_manager.py
├── test_transforms_mem.py
└── test_policy_mem.py
```

---

## 新增文件

| 文件 | 说明 |
|------|------|
| `high_level_policy.py` | 高层策略：子任务调度和语言记忆更新 |
| `memory_manager.py` | 语言记忆管理器：生成、更新、压缩、序列化 |
| `gen_memory_labels.py` | 离线语言记忆标注生成脚本 |

## 修改文件

| 文件 | 改动 |
|------|------|
| `siglip.py` | 添加 SpaceTimeSeparableBlock |
| `model.py` | 添加 image_history, tokenized_memory 等字段 |
| `pi0.py` | 添加 MEM 前向传播逻辑 |
| `pi0_config.py` | 添加 MEMConfig 配置类 |
| `policy.py` | 添加 MEMPolicy 类 |
| `transforms.py` | 添加 VideoFrameStack, TokenizeMemory |
| `data_loader.py` | 添加 MEMLeRobotDataset |

---

## 配置说明

### MEMConfig

```python
@dataclass
class MEMConfig:
    # 短期视觉记忆
    use_video_memory: bool = False          # 是否启用视频记忆
    video_memory_frames: int = 6            # 历史帧数 K
    temporal_attn_every_n_layers: int = 4   # 每隔几层插入时间注意力
    drop_history_tokens_after_layer: int = -4  # 在第几层后丢弃历史帧

    # 长期语言记忆
    use_language_memory: bool = False       # 是否启用语言记忆
    max_memory_tokens: int = 256            # 最大 token 数
    memory_loss_weight: float = 0.1          # 记忆损失权重

    # 本体感觉历史
    use_state_history: bool = False          # 是否启用状态历史
    state_history_frames: int = 6            # 状态历史帧数
```

---

## 约束条件

| 约束 | 说明 |
|------|------|
| **向后兼容** | `image_history=None` 时所有 MEM 分支跳过，行为等同 π0.5 |
| **零新参数** | 视频编码器仅修改注意力模式，不新增可学习参数 |
| **K=1 数值等价** | 单帧时输出与原 SigLIP ViT 数值一致（误差 < 1e-5） |
| **实时性** | 6 帧时推理延迟 < 300ms（H100 单卡，4 路摄像头） |

---

## 测试

运行测试：
```bash
# 视频编码器测试
python -m pytest tests/test_video_encoder.py -v

# MEM 模型测试
python -m pytest tests/test_mem_model.py -v

# 记忆管理器测试
python -m pytest tests/test_memory_manager.py -v

# 推理策略测试
python -m pytest tests/test_policy_mem.py -v

# 数据变换测试
python -m pytest tests/test_transforms_mem.py -v
```

---

## 参考资料

- 原始论文：[MEM: Multi-Scale Embodied Memory for Vision Language Action Models](https://pi.website/research/memory)
- 基础代码：[Physical Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

---

## License

继承自 openpi 项目的 Apache 2.0 许可证
