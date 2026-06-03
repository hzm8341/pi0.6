# openpi (pi0.6 Custom Version)

openpi holds open-source models and packages for robotics, published by the [Physical Intelligence team](https://www.physicalintelligence.company/).

This repository is a custom implementation based on π₀.₅ and π₀.₆ papers, modified for specific research purposes.

Currently, this repo contains three types of models:
- the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based vision-language-action model (VLA).
- the [π₀-FAST model](https://www.physicalintelligence.company/research/fast), an autoregressive VLA, based on the FAST action tokenizer.
- the [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05), an upgraded version of π₀ with better open-world generalization trained with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation). Note that, in this repository, we currently only support the flow matching head for both $\pi_{0.5}$ training and inference.

For all models, we provide _base model_ checkpoints, pre-trained on 10k+ hours of robot data, and examples for using them out of the box or fine-tuning them to your own datasets.

This is an experiment: $\pi_0$ was developed for our own robots, which differ from the widely used platforms such as [ALOHA](https://tonyzhaozh.github.io/aloha/) and [DROID](https://droid-dataset.github.io/), and though we are optimistic that researchers and practitioners will be able to run creative new experiments adapting $\pi_0$ to their own platforms, we do not expect every such attempt to be successful. All this is to say: $\pi_0$ may or may not work for you, but you are welcome to try it and see!

## 本 fork 已实现的核心功能 / Implemented features in this fork

中文：这个 fork 目前优先实现了两条研究线：一条是 $\pi_{0.5}$ / $\pi_{0.6}$ 兼容的 MEM/G1 微调工作流，另一条是面向 $\pi_{0.6}$ RECAP 思路的优势条件化训练基础设施。已落地的 RECAP 代码是可继续扩展的实验框架，不是完整论文复现。

English: This fork currently focuses on two research tracks: a MEM/G1 fine-tuning workflow compatible with $\pi_{0.5}$ / $\pi_{0.6}$, and an advantage-conditioned training infrastructure inspired by the $\pi_{0.6}$ RECAP method. The implemented RECAP code is an extensible experimental framework, not a complete paper reproduction.

### 已实现 / Implemented

| 功能 / Feature | 状态 / Status | 主要代码位置 / Main code location |
| -------------- | ------------- | --------------------------------- |
| RECAP 配置项 / RECAP config | 已实现 / Implemented | `src/openpi/models/pi0_config.py:ReCAPConfig` |
| 观测数据中携带优势标签、人类干预标签 / Advantage and human-intervention fields in observations | 已实现 / Implemented | `src/openpi/models/model.py:Observation` |
| advantage prompt 预 tokenization / Pre-tokenization for advantage prompts | 已实现 / Implemented | `src/openpi/transforms.py:TokenizeReCAPAdvantage` |
| 在 `Pi0.embed_prefix()` 注入 `Advantage: positive/negative` tokens / Inject `Advantage: positive/negative` tokens in `Pi0.embed_prefix()` | 已实现 / Implemented | `src/openpi/models/pi0.py` |
| RECAP 双路 loss 路由 / Two-branch RECAP loss route | 已实现 / Implemented | `src/openpi/models/pi0.py:compute_recap_loss()`、`scripts/train.py` |
| RECAP 训练配置注册 / RECAP train config registration | 已实现 / Implemented | `src/openpi/training/config.py` 中的 / entries: `debug_pi05_recap`、`pi05_recap`、`pi05_aloha_recap` |
| 回报、return bin、advantage 计算工具 / Reward, return-bin, and advantage utilities | 已实现 / Implemented | `src/openpi/models/value_function.py` |
| episode advantage 标注工具 / Episode advantage-labeling utilities | 已实现 / Implemented | `src/openpi/training/recap_collector.py` |
| 离线 RECAP episode JSON 读写 / Offline RECAP episode JSON IO | 已实现 / Implemented | `src/openpi/training/recap_episode_io.py` |
| 离线 reward/return/value proxy/advantage 生成 / Offline reward, return, value-proxy, and advantage generation | 已实现基础版 / Basic implementation complete | `src/openpi/training/recap_value_proxy.py`、`src/openpi/training/recap_offline.py` |
| RECAP 标签侧车文件导出 / RECAP label sidecar export | 已实现 / Implemented | `scripts/label_recap_advantage.py` 输出 `recap_labels.jsonl` 和 `lerobot_fields.npz` |
| RECAP 侧车字段接入 dataloader / RECAP sidecar field merge in dataloader | 已实现 / Implemented | `src/openpi/training/data_loader.py:ReCAPFieldsDataset`、`DataConfig.recap_fields_path` |
| RECAP split/trim/eval 数据准备工具 / RECAP split, trim, and eval preparation tools | 已实现 / Implemented | `scripts/create_recap_split_manifest.py`、`scripts/trim_recap_episodes.py`、`scripts/eval_recap_episodes.py` |
| RECAP 迭代训练 CLI 骨架 / RECAP iterative-training CLI skeleton | 已实现 / Implemented | `scripts/recap_train.py` |
| 未完成项跟踪 / Remaining work tracking | 已实现 / Implemented | `TODO_RECAP.md` |
| 实施计划和阶段拆解 / Implementation plan and phase breakdown | 已实现 / Implemented | `docs/superpowers/plans/2026-06-02-recap.md` |

### 对比 $\pi_{0.6}$ / RECAP 论文：已完成与未完成 / Paper comparison: completed vs. missing

| 论文模块 / Paper module | 当前状态 / Current status | 说明 / Notes |
| ----------------------- | ------------------------- | ------------ |
| 优势条件化策略输入 / Advantage-conditioned policy input | 已完成基础实现 / Basic implementation complete | 训练样本可以携带 `advantage_indicator`，模型 prefix 中会注入 positive/negative advantage tokens。 / Training samples can carry `advantage_indicator`, and the model prefix injects positive/negative advantage tokens. |
| RECAP 策略 loss / RECAP policy loss | 已完成基础实现 / Basic implementation complete | `compute_recap_loss()` 同时计算无条件 loss 和有条件 loss，并支持 advantage dropout。 / `compute_recap_loss()` computes both unconditional and conditioned losses and supports advantage dropout. |
| 使用 `use_advantage: bool[batch]` mask 处理条件 dropout / Conditional dropout via `use_advantage: bool[batch]` mask | 已完成 / Implemented | 避免在 JIT 内动态生成 `None`。 / Avoids dynamically generating `None` inside JIT. |
| 分布式价值函数的奖励/return/bin 工具 / Reward, return, and bin utilities for distributional value learning | 部分完成 / Partially complete | 已有工具函数、轻量 value head scaffold、离线 progress value proxy；完整视觉语言 value backbone 还没有接入。 / Utility functions, a lightweight value-head scaffold, and an offline progress value proxy exist; the full vision-language value backbone is not wired yet. |
| 根据 value/value proxy 计算优势并生成训练标签 / Compute advantages from value/value proxy and generate labels | 基础离线版完成 / Basic offline version complete | `scripts/label_recap_advantage.py` 可从 episode JSON 生成侧车字段，`recap_fields_path` 可在 dataloader 中按 frame 顺序合并；真实 LeRobot 原地写回仍在 TODO。 / `scripts/label_recap_advantage.py` can generate sidecar fields, and `recap_fields_path` can merge them in the dataloader by frame order; true in-place LeRobot write-back remains TODO. |
| 人类在环数据采集 / Human-in-the-loop data collection | 部分完成 / Partially complete | 数据结构保留 `is_human_intervention`，collector 有 episode/label 工具；还没有接入真实机器人环境、SpaceMouse/WebSocket 等干预回调。 / Data structures preserve `is_human_intervention`, and collector episode/label utilities exist; real robot environments and SpaceMouse/WebSocket intervention callbacks are not wired yet. |
| Algorithm 1 迭代循环：collect -> train value -> label advantage -> finetune VLA / Algorithm 1 loop: collect -> train value -> label advantage -> finetune VLA | 骨架完成 / Skeleton complete | `scripts/recap_train.py` 记录阶段顺序；还没有自动调用完整 value training、VLA training 和机器人 rollout。 / `scripts/recap_train.py` records the stage order; it does not yet automatically launch full value training, VLA training, or robot rollouts. |
| 每轮从预训练 checkpoint 重新微调以降低策略漂移 / Restart each iteration from the pretrained checkpoint to reduce policy drift | 设计已记录 / Design recorded | README/计划中记录了训练方式；脚本还需要实际 checkpoint orchestration。 / The training design is documented in README/plan; actual checkpoint orchestration still needs implementation. |
| 在线/真实机器人 HITL RL 闭环 / Online real-robot HITL RL loop | 未完成 / Not complete | 需要实现 env adapter、intervention callback、episode 存储格式、成功/失败标注入口和安全停机逻辑。 / Requires env adapters, intervention callbacks, episode storage, success/failure labeling, and safety-stop logic. |

### 如何训练 RECAP VLA / How to train a RECAP VLA

中文：当前可训练的是 RECAP-enabled VLA 监督/离线微调路径。你的 LeRobot 或自定义数据需要在 transform 之后包含这些字段：

English: The currently trainable path is RECAP-enabled supervised/offline VLA fine-tuning. Your LeRobot or custom dataset should contain these fields after transforms:

- `advantage_indicator`: `bool`，表示当前样本动作优势是否为正。 / Whether the current sample has positive action advantage.
- `use_advantage`: `bool`，表示本样本是否启用 advantage condition。 / Whether this sample enables the advantage condition.
- `is_human_intervention`: `bool`，表示样本是否来自人类干预动作。 / Whether this sample came from human intervention.
- `tokenized_advantage_positive`、`tokenized_advantage_negative`、`tokenized_advantage_mask`: 通常由 `TokenizeReCAPAdvantage` 自动生成。 / Usually generated automatically by `TokenizeReCAPAdvantage`.

训练命令 / Training commands:

```bash
# 1. 计算归一化统计量。 / Compute normalization statistics.
uv run scripts/compute_norm_stats.py --config-name pi05_recap

# 2. 运行 RECAP-enabled VLA 训练。 / Run RECAP-enabled VLA training.
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_recap \
  --exp-name=my_recap_experiment \
  --overwrite

# ALOHA 数据可使用： / For ALOHA data:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_aloha_recap \
  --exp-name=my_aloha_recap_experiment \
  --overwrite

# 3. 调试配置，用 dummy model 做最小 smoke run。 / Debug config with a dummy model.
uv run scripts/train.py debug_pi05_recap \
  --exp-name=recap_debug \
  --num-train-steps=1 \
  --batch-size=2 \
  --overwrite \
  --wandb-enabled=False
```

离线迭代训练骨架 / Offline iterative-training skeleton:

```bash
PYTHONPATH=src python scripts/recap_train.py \
  --task-name my_task \
  --demo-dataset /path/to/demo_episodes \
  --num-iterations 3
```

中文：这个 CLI 目前用于组织 RECAP 迭代阶段，还不会自动完成真实 rollout、value model 训练和 VLA 训练调用。

English: This CLI currently organizes the RECAP iteration stages. It does not yet automatically run real rollouts, value-model training, or VLA training calls.

### 离线 RECAP 标签生成 / Offline RECAP label generation

中文：当前已经可以先跑通离线版 RECAP 的前半段：读取人工标注 success/failure 的 episode JSON，生成 reward、return、progress value proxy、n-step advantage，并输出可合并到 LeRobot 数据集的侧车字段。

English: The first half of offline RECAP is available now: load manually labeled success/failure episode JSON, generate rewards, returns, a progress value proxy, n-step advantages, and export sidecar fields that can be merged into a LeRobot dataset.

episode JSON 示例 / Example episode JSON:

```json
{
  "episode_id": "ep001",
  "task": "pick test tube rack and insert into slot",
  "success": true,
  "timeout": false,
  "max_episode_length": 500,
  "frames": [
    {
      "t": 0,
      "observation": {"state": [0.0, 0.1]},
      "action": [0.01, -0.02],
      "is_human_intervention": false
    }
  ],
  "metadata": {"source": "demo"}
}
```

生成标签 / Generate labels:

```bash
PYTHONPATH=src python scripts/label_recap_advantage.py \
  --input-episodes /path/to/recap_episode_json_or_dir \
  --output-dir outputs/recap_labels \
  --positive-fraction 0.4 \
  --n-step-lookahead 50
```

训练前数据准备 / Pre-training data preparation:

```bash
# 1. 生成 train/eval manifest，避免 episode 泄漏。
# Create a train/eval manifest to avoid episode leakage.
PYTHONPATH=src python scripts/create_recap_split_manifest.py \
  --input-episodes /path/to/recap_episode_json_or_dir \
  --output outputs/recap_labels/split_manifest.json \
  --eval-fraction 0.15 \
  --seed 0

# 2. 裁剪尾部静止 episode。 / Trim static tail frames.
PYTHONPATH=src python scripts/trim_recap_episodes.py \
  --input-episodes /path/to/recap_episode_json_or_dir \
  --output-episodes outputs/recap_trimmed_episodes \
  --action-norm-threshold 0.05 \
  --min-frames 10

# 3. 输出离线数据摘要。 / Write an offline data summary.
PYTHONPATH=src python scripts/eval_recap_episodes.py \
  --input-episodes outputs/recap_trimmed_episodes \
  --output outputs/recap_labels/offline_eval_summary.json \
  --static-action-threshold 0.05
```

输出 / Outputs:

- `outputs/recap_labels/recap_labels.jsonl`: 每个 timestep 的 `reward`、`return`、`value`、`advantage`、`advantage_indicator`、`label_source`。
- `outputs/recap_labels/lerobot_fields.npz`: 扁平数组 `advantage_indicator`、`use_advantage`、`is_human_intervention`，用于后续合并到 LeRobot 数据表。

GPU 服务器训练时，需要把 `lerobot_fields.npz` 作为 `DataConfig.recap_fields_path` 传入训练配置；该文件长度必须和 LeRobot flattened frame dataset 长度一致。若使用 tyro CLI，可尝试：

```bash
uv run scripts/train.py pi05_recap \
  --data.recap-fields-path outputs/recap_labels/lerobot_fields.npz \
  --exp-name=my_recap_experiment \
  --overwrite
```

也可以直接使用 GPU 服务器验证脚本。详细步骤见 `docs/recap_gpu_training.md`：

```bash
EPISODES_DIR=/path/to/recap_episode_json_or_dir \
CONFIG_NAME=pi05_recap \
EXP_NAME=test_tube_recap_v1 \
WANDB_ENABLED=False \
bash scripts/run_recap_gpu_validation.sh
```

中文：这里的 value 是 progress-based proxy，不是论文中的完整视觉语言 distributional value function。真实 LeRobot 原地写回、轻量可训练 value model、完整 value model、真机 rollout 和 HITL callback 仍记录在 `TODO_RECAP.md`。

English: The value here is a progress-based proxy, not the full vision-language distributional value function from the paper. True in-place LeRobot write-back, a trainable lightweight value model, the full value model, real rollouts, and HITL callbacks remain tracked in `TODO_RECAP.md`.

### RECAP TODO / Remaining RECAP work

中文：以下是当前还不能在本仓库内完整实现、或需要真实数据/机器人接口后才能完成的 RECAP 项目。内容与 `TODO_RECAP.md` 保持一致。

English: The following RECAP items are not fully implemented in this repository yet, or require real datasets / robot interfaces before they can be completed. This mirrors `TODO_RECAP.md`.

#### Offline RECAP

- [ ] Implement true LeRobot in-place dataset write-back.
  - Current status: `scripts/label_recap_advantage.py` writes sidecar files: `recap_labels.jsonl` and `lerobot_fields.npz`; `DataConfig.recap_fields_path` can merge `lerobot_fields.npz` into training samples at dataloader time.
  - Needed: optional permanent integration into the actual LeRobot dataset frame table or metadata format used by the target dataset.
- [ ] Train a learned lightweight value model.
  - Current status: offline labeling uses a deterministic progress-based value proxy.
  - Needed: add a small trainable value model that consumes state and/or image embeddings, trains with return targets, saves checkpoints, and supports batched inference.
- [ ] Add an evaluation harness comparing `pi05_recap` against normal `pi05` SFT.
  - Needed metrics: success rate, average completion time, failure type counts, and label distribution.
- [ ] Connect `scripts/recap_train.py` to real subprocesses.
  - Current status: it logs the RECAP stages.
  - Needed: call label generation, VLA training, evaluation, and report generation.

#### Rollout and HITL

- [ ] Implement `src/openpi/rollout/recap_env.py`.
  - Interface: `reset()`, `step(action)`, `success()`, `close()`.
- [ ] Implement `src/openpi/rollout/episode_recorder.py`.
  - Record observation, policy action, human action, executed action, success/failure, timeout, and intervention flags.
- [ ] Implement `src/openpi/rollout/intervention.py`.
  - Support keyboard, SpaceMouse, WebSocket, or platform-specific teleoperation callbacks.
- [ ] Implement `src/openpi/rollout/safety.py`.
  - Add velocity limits, joint/workspace limits, force thresholds, action smoothing, timeout stop, and emergency stop hooks.
- [ ] Implement success/failure labelers for the target task.
  - Start with manual labels, then add task-specific automatic checks where reliable.

#### Paper-Level π0.6

- [ ] Upgrade the VLA backbone toward paper-level π0.6 scale.
  - Target: Gemma 3 4B base VLM and larger action expert.
  - This requires checkpoint availability, memory planning, training config changes, and validation on the target hardware.
- [ ] Add full KI joint objective coverage for RECAP.
  - Current status: RECAP loss wraps flow matching only.
  - Needed: include sub-task text prediction and FAST/discrete action likelihood where the model path supports them.
- [ ] Implement the full vision-language distributional value function.
  - Target: image + robot state + language input, 201 value bins, cross-entropy training, checkpoint save/load, and inference.

### $\pi_{0.6}$ 的人类在环强化学习部分如何训练 / How to train the human-in-the-loop RL part of $\pi_{0.6}$

中文：RECAP / $\pi_{0.6}$ 中的人类在环强化学习可以理解为离线迭代的人类辅助 RL 流程：策略先自主 rollout；人在失败、危险或低质量动作时接管；系统记录接管前后的 episode、成功/失败和人类动作；随后训练 value function，计算每个状态动作的优势，并把优势正负作为条件 token 继续微调 VLA。

English: The human-in-the-loop RL part of RECAP / $\pi_{0.6}$ can be understood as an offline iterative human-assisted RL workflow: the policy first performs autonomous rollouts; a human takes over on failures, dangerous states, or low-quality actions; the system records episodes, success/failure outcomes, and human actions; then a value function is trained, action advantages are computed, and the VLA is fine-tuned with positive/negative advantage tokens as conditions.

当前代码已经提供这些入口 / Current code entry points:

- `src/openpi/training/recap_collector.py`: episode 数据结构、优势标签分配。 / Episode data structures and advantage-label assignment.
- `src/openpi/models/value_function.py`: reward、return、advantage、bin 工具。 / Reward, return, advantage, and bin utilities.
- `src/openpi/models/model.py`: `is_human_intervention`、`advantage_indicator` 字段。 / `is_human_intervention` and `advantage_indicator` fields.
- `src/openpi/models/pi0.py`: advantage-conditioned 策略训练。 / Advantage-conditioned policy training.
- `scripts/recap_train.py`: 迭代训练流程骨架。 / Iterative training flow skeleton.

要训练完整 HITL RL，还需要补齐以下项目特定代码 / Full HITL RL training still requires these project-specific pieces:

1. 机器人或仿真环境 adapter：提供 `reset()`、`step(action)`、`success`/`done` 标记。 / Robot or simulation env adapter exposing `reset()`, `step(action)`, and `success`/`done` labels.
2. 人类干预 callback：例如 SpaceMouse、键盘、WebSocket 或遥操作手柄；无干预时返回 `None`，有干预时返回人类动作。 / Human-intervention callback, such as SpaceMouse, keyboard, WebSocket, or teleoperation controller; return `None` when there is no intervention and a human action when intervention occurs.
3. episode 存储格式：建议用结构化 `npz`/`jsonl`/`parquet`，不要用不可信 pickle。 / Episode storage format; use structured `npz`/`jsonl`/`parquet`, not untrusted pickle.
4. value function 训练循环：把已采集 episode 转成 value targets，训练完整视觉语言 value model。 / Value-function training loop that converts collected episodes into value targets and trains the full vision-language value model.
5. advantage 写回数据集：当前可生成侧车文件；真实 LeRobot 原地写回仍在 `TODO_RECAP.md`。 / Dataset write-back: sidecar generation is available now; true in-place LeRobot write-back remains in `TODO_RECAP.md`.
6. VLA 微调：使用 `pi05_recap` 或平台专用 RECAP config 重新从 base checkpoint fine-tune。 / VLA fine-tuning from the base checkpoint using `pi05_recap` or a platform-specific RECAP config.
7. 重复迭代：collect -> train value -> label advantage -> finetune VLA。 / Repeat the iteration: collect -> train value -> label advantage -> finetune VLA.

中文：换句话说，当前仓库已经实现了 RECAP VLA 训练所需的模型输入、loss、配置和离线工具；真实“人在环强化学习闭环”还需要接入具体机器人平台和 value model 训练 orchestration。

English: In short, this repository already implements the model inputs, loss, configs, and offline utilities needed for RECAP VLA training; a real human-in-the-loop RL closed loop still needs platform-specific robot integration and value-model training orchestration.

## Updates

- [Sept 2025] We released PyTorch support in openpi.
- [Sept 2025] We released pi05, an upgraded version of pi0 with better open-world generalization.
- [Sept 2025]: We have added an [improved idle filter](examples/droid/README_train.md#data-filtering) for DROID training.
- [Jun 2025]: We have added [instructions](examples/droid/README_train.md) for using `openpi` to train VLAs on the full [DROID dataset](https://droid-dataset.github.io/). This is an approximate open-source implementation of the training pipeline used to train pi0-FAST-DROID.
- This fork adds an experimental RECAP implementation path for $\pi_{0.5}$ / $\pi_{0.6}$ advantage-conditioned training (see [RECAP advantage-conditioned training](#recap-advantage-conditioned-training) below).
- This fork adds π₀.₅ fine-tuning presets and scripts for the **Unitree G1** LeRobot dataset *g1-pick-apple* (see [G1 humanoid (pick-apple) fine-tuning](#g1-humanoid-pick-apple-fine-tuning) below).


## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.

## Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

**Docker**: As an alternative to uv installation, we provide instructions for installing openpi using Docker. If you encounter issues with your system setup, consider using Docker to simplify installation. See [Docker Setup](docs/docker.md) for more details.

This fork also ships a CUDA 12.4 training image and helper scripts at the repo root: [`Dockerfile`](Dockerfile), [`setup_env.sh`](setup_env.sh), [`train_g1_pick_apple.sh`](train_g1_pick_apple.sh), and [`run_train_openpi_image.sh`](run_train_openpi_image.sh). Use them when the upstream Docker docs do not match your G1 workflow.

### G1 humanoid (pick-apple) fine-tuning

Fine-tuning presets target the LeRobot dataset [`PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-apple`](https://huggingface.co/datasets/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-apple) (43-DOF state/action, ego-view camera). Policy transforms live in [`src/openpi/policies/g1_policy.py`](src/openpi/policies/g1_policy.py).

| Config | GPU memory (typical) | Use case |
| ------ | -------------------- | -------- |
| `pi05_g1_pick_apple_lora` | ~24 GB | Default in `train_g1_pick_apple.sh`; LoRA on π₀.₅ |
| `pi05_g1_pick_apple` | > 70 GB | Full fine-tuning (e.g. A100 / H100) |

**Host requirements:** Native `uv sync` needs **glibc ≥ 2.31** (many prebuilt wheels). On older hosts, build and run with the root [`Dockerfile`](Dockerfile).

**Data layout:** Set `HF_LEROBOT_HOME` (or `DATASET_ROOT` for the train script) to the directory that contains your LeRobot-style dataset tree. The default is `./datasets`; that path is listed in `.gitignore` so local data is not committed.

**Train:**

```bash
# Optional one-shot env install (uv + deps; see script for checkpoint download)
bash setup_env.sh

export DATASET_ROOT=/path/to/your/datasets   # optional; default: <repo>/datasets
bash train_g1_pick_apple.sh my_experiment_name
```

Override the preset with `CONFIG=pi05_g1_pick_apple` or `CONFIG=pi05_g1_pick_apple_lora`. If a checkpoint directory for that run already exists, the script continues with **`--resume`** unless you set **`OVERWRITE=1`** (fresh run) or **`RESUME=1`** explicitly.

**Docker (existing openpi image):**

```bash
OPENPI_DOCKER_IMAGE=your:image bash run_train_openpi_image.sh my_experiment_name
```


### RECAP advantage-conditioned training

RECAP functionality is summarized near the top of this README in [本 fork 已实现的核心功能](#本-fork-已实现的核心功能), including implemented modules, paper comparison, training commands, human-in-the-loop RL status, and code locations.


## Model Checkpoints

### Base Models
We provide multiple base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data, and can be used for fine-tuning.

| Model        | Use Case    | Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | Fine-Tuning | Base [π₀ model](https://www.physicalintelligence.company/blog/pi0) for fine-tuning                | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | Fine-Tuning | Base autoregressive [π₀-FAST model](https://www.physicalintelligence.company/research/fast) for fine-tuning | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$    | Fine-Tuning | Base [π₀.₅ model](https://www.physicalintelligence.company/blog/pi05) for fine-tuning    | `gs://openpi-assets/checkpoints/pi05_base`      |

### Fine-Tuned Models
We also provide "expert" checkpoints for various robot platforms and tasks. These models are fine-tuned from the base models above and intended to run directly on the target robot. These may or may not work on your particular robot. Since these checkpoints were fine-tuned on relatively small datasets collected with more widely available robots, such as ALOHA and the DROID Franka setup, they might not generalize to your particular setup, though we found some of these, especially the DROID checkpoint, to generalize quite broadly in practice.

| Model                    | Use Case    | Description                                                                                                                                                                                              | Checkpoint Path                                       |
| ------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| $\pi_0$-FAST-DROID       | Inference   | $\pi_0$-FAST model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): can perform a wide range of simple table-top manipulation tasks 0-shot in new scenes on the DROID robot platform | `gs://openpi-assets/checkpoints/pi0_fast_droid`       |
| $\pi_0$-DROID            | Fine-Tuning | $\pi_0$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/): faster inference than $\pi_0$-FAST-DROID, but may not follow language commands as well                                | `gs://openpi-assets/checkpoints/pi0_droid`            |
| $\pi_0$-ALOHA-towel      | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can fold diverse towels 0-shot on ALOHA robot platforms                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_towel`      |
| $\pi_0$-ALOHA-tupperware | Inference   | $\pi_0$ model fine-tuned on internal [ALOHA](https://tonyzhaozh.github.io/aloha/) data: can unpack food from a tupperware container                                                                                                             | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap  | Inference   | $\pi_0$ model fine-tuned on public [ALOHA](https://dit-policy.github.io/) data: can uncap a pen                                                                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |
| $\pi_{0.5}$-LIBERO      | Inference   | $\pi_{0.5}$ model fine-tuned for the [LIBERO](https://libero-project.github.io/datasets) benchmark: gets state-of-the-art performance (see [LIBERO README](examples/libero/README.md)) | `gs://openpi-assets/checkpoints/pi05_libero`      |
| $\pi_{0.5}$-DROID      | Inference / Fine-Tuning | $\pi_{0.5}$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/) with [knowledge insulation](https://www.physicalintelligence.company/research/knowledge_insulation): fast inference and good language-following | `gs://openpi-assets/checkpoints/pi05_droid`      |


By default, checkpoints are automatically downloaded from `gs://openpi-assets` and are cached in `~/.cache/openpi` when needed. You can overwrite the download path by setting the `OPENPI_DATA_HOME` environment variable.




## Running Inference for a Pre-Trained Model

Our pre-trained model checkpoints can be run with a few lines of code (here our $\pi_0$-FAST-DROID model):
```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```
You can also test this out in the [example notebook](examples/inference.ipynb).

We provide detailed step-by-step examples for running inference of our pre-trained checkpoints on [DROID](examples/droid/README.md) and [ALOHA](examples/aloha_real/README.md) robots.

**Remote Inference**: We provide [examples and code](docs/remote_inference.md) for running inference of our models **remotely**: the model can run on a different server and stream actions to the robot via a websocket connection. This makes it easy to use more powerful GPUs off-robot and keep robot and policy environments separate.

**Test inference without a robot**: We provide a [script](examples/simple_client/README.md) for testing inference without a robot. This script will generate a random observation and run inference with the model. See [here](examples/simple_client/README.md) for more details.





## Fine-Tuning Base Models on Your Own Data

We will fine-tune the $\pi_{0.5}$ model on the [LIBERO dataset](https://libero-project.github.io/datasets) as a running example for how to fine-tune a base model on your own data. We will explain three steps:
1. Convert your data to a LeRobot dataset (which we use for training)
2. Defining training configs and running training
3. Spinning up a policy server and running inference

### 1. Convert your data to a LeRobot dataset

We provide a minimal example script for converting LIBERO data to a LeRobot dataset in [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py). You can easily modify it to convert your own data! You can download the raw LIBERO dataset from [here](https://huggingface.co/datasets/openvla/modified_libero_rlds), and run the script with:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**Note:** If you just want to fine-tune on LIBERO, you can skip this step, because our LIBERO fine-tuning configs point to a pre-converted LIBERO dataset. This step is merely an example that you can adapt to your own data.

### 2. Defining training configs and running training

To fine-tune a base model on your own data, you need to define configs for data processing and training. We provide example configs with detailed comments for LIBERO below, which you can modify for your own dataset:

- [`LiberoInputs` and `LiberoOutputs`](src/openpi/policies/libero_policy.py): Defines the data mapping from the LIBERO environment to the model and vice versa. Will be used for both, training and inference.
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py): Defines how to process raw LIBERO data from LeRobot dataset for training.
- [`TrainConfig`](src/openpi/training/config.py): Defines fine-tuning hyperparameters, data config, and weight loader.

We provide example fine-tuning configs for [π₀](src/openpi/training/config.py), [π₀-FAST](src/openpi/training/config.py), and [π₀.₅](src/openpi/training/config.py) on LIBERO data.

Before we can run training, we need to compute the normalization statistics for the training data. Run the script below with the name of your training config:

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

Now we can kick off training with the following command (the `--overwrite` flag is used to overwrite existing checkpoints if you rerun fine-tuning with the same config):

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

The command will log training progress to the console and save checkpoints to the `checkpoints` directory. You can also monitor training progress on the Weights & Biases dashboard. For maximally using the GPU memory, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before running training -- this enables JAX to use up to 90% of the GPU memory (vs. the default of 75%).

**Note:** We provide functionality for *reloading* normalization statistics for state / action normalization from pre-training. This can be beneficial if you are fine-tuning to a new task on a robot that was part of our pre-training mixture. For more details on how to reload normalization statistics, see the [norm_stats.md](docs/norm_stats.md) file.

### 3. Spinning up a policy server and running inference

Once training is complete, we can run inference by spinning up a policy server and then querying it from a LIBERO evaluation script. Launching a model server is easy (we use the checkpoint for iteration 20,000 for this example, modify as needed):

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

This will spin up a server that listens on port 8000 and waits for observations to be sent to it. We can then run an evaluation script (or robot runtime) that queries the server.

For running the LIBERO eval in particular, we provide (and recommend using) a Dockerized workflow that handles both the policy server and the evaluation script together. See the [LIBERO README](examples/libero/README.md) for more details.

If you want to embed a policy server call in your own robot runtime, we have a minimal example of how to do so in the [remote inference docs](docs/remote_inference.md).



### More Examples

We provide more examples for how to fine-tune and run inference with our models on the ALOHA platform in the following READMEs:
- [ALOHA Simulator](examples/aloha_sim)
- [ALOHA Real](examples/aloha_real)
- [UR5](examples/ur5)

## PyTorch Support

openpi now provides PyTorch implementations of π₀ and π₀.₅ models alongside the original JAX versions! The PyTorch implementation has been validated on the LIBERO benchmark (both inference and finetuning). A few features are currently not supported (this may change in the future):

- The π₀-FAST model
- Mixed precision training
- FSDP (fully-sharded data parallelism) training
- LoRA (low-rank adaptation) training
- EMA (exponential moving average) weights during training

### Setup
1. Make sure that you have the latest version of all dependencies installed: `uv sync`

2. Double check that you have transformers 4.53.2 installed: `uv pip show transformers`

3. Apply the transformers library patches:
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```

This overwrites several files in the transformers library with necessary model changes: 1) supporting AdaRMS, 2) correctly controlling the precision of activations, and 3) allowing the KV cache to be used without being updated.

**WARNING**: With the default uv link mode (hardlink), this will permanently affect the transformers library in your uv cache, meaning the changes will survive reinstallations of transformers and could even propagate to other projects that use transformers. To fully undo this operation, you must run `uv cache clean transformers`.

### Converting JAX Models to PyTorch

To convert a JAX model checkpoint to PyTorch format:

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### Running Inference with PyTorch

The PyTorch implementation uses the same API as the JAX version - you only need to change the checkpoint path to point to the converted PyTorch model:

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# Create a trained policy (automatically detects PyTorch format)
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference (same API as JAX)
action_chunk = policy.infer(example)["actions"]
```

### Policy Server with PyTorch

The policy server works identically with PyTorch models - just point to the converted checkpoint directory:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### Finetuning with PyTorch

To finetune a model in PyTorch:

1. Convert the JAX base model to PyTorch format:
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. Specify the converted PyTorch model path in your config using `pytorch_weight_path`

3. Launch training using one of these modes:

```bash
# Single GPU training:
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# Example:
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # Resume from latest checkpoint

# Multi-GPU training (single node):
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# Example:
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# Multi-Node Training:
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### Precision Settings

JAX and PyTorch implementations handle precision as follows:

**JAX:**
1. Inference: most weights and computations in bfloat16, with a few computations in float32 for stability
2. Training: defaults to mixed precision: weights and gradients in float32, (most) activations and computations in bfloat16. You can change to full float32 training by setting `dtype` to float32 in the config.

**PyTorch:**
1. Inference: matches JAX -- most weights and computations in bfloat16, with a few weights converted to float32 for stability
2. Training: supports either full bfloat16 (default) or full float32. You can change it by setting `pytorch_training_precision` in the config. bfloat16 uses less memory but exhibits higher losses compared to float32. Mixed precision is not yet supported.

With torch.compile, inference speed is comparable between JAX and PyTorch.

## Troubleshooting

We will collect common issues and their solutions here. If you encounter an issue, please check here first. If you can't find a solution, please file an issue on the repo (see [here](CONTRIBUTING.md) for guidelines).

| Issue                                     | Resolution                                                                                                                                                                                   |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync` fails with dependency conflicts | Try removing the virtual environment directory (`rm -rf .venv`) and running `uv sync` again. If issues persist, check that you have the latest version of `uv` installed (`uv self update`). |
| Training runs out of GPU memory           | Make sure you set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (or higher) before running training to allow JAX to use more GPU memory. You can also use `--fsdp-devices <n>` where `<n>` is your number of GPUs, to enable [fully-sharded data parallelism](https://engineering.fb.com/2021/07/15/open-source/fsdp/), which reduces memory usage in exchange for slower training (the amount of slowdown depends on your particular setup). If you are still running out of memory, you may want to consider disabling EMA.        |
| Policy server connection errors           | Check that the server is running and listening on the expected port. Verify network connectivity and firewall settings between client and server.                                            |
| Missing norm stats error when training    | Run `scripts/compute_norm_stats.py` with your config name before starting training.                                                                                                          |
| Dataset download fails                    | Check your internet connection. For HuggingFace datasets, ensure you're logged in (`huggingface-cli login`).                                                                                 |
| CUDA/GPU errors                           | Verify NVIDIA drivers are installed correctly. For Docker, ensure nvidia-container-toolkit is installed. Check GPU compatibility. You do NOT need CUDA libraries installed at a system level --- they will be installed via uv. You may even want to try *uninstalling* system CUDA libraries if you run into CUDA issues, since system libraries can sometimes cause conflicts. |
| Import errors when running examples       | Make sure you've installed all dependencies with `uv sync`. Some examples may have additional requirements listed in their READMEs.                    |
| Action dimensions mismatch                | Verify your data processing transforms match the expected input/output dimensions of your robot. Check the action space definitions in your policy classes.                                  |
| Diverging training loss                            | Check the `q01`, `q99`, and `std` values in `norm_stats.json` for your dataset. Certain dimensions that are rarely used can end up with very small `q01`, `q99`, or `std` values, leading to huge states and actions after normalization. You can manually adjust the norm stats as a workaround. |

---

# π0.6-MEM: Multi-Scale Embodied Memory for Vision-Language-Action Models

> Multi-Scale Embodied Memory (MEM) architecture implemented on top of the openpi (π0.5) codebase
> Reference Paper: [MEM: Multi-Scale Embodied Memory for Vision Language Action Models](https://pi.website/research/memory)

---

## Overview

π0.6-MEM is an extended version of Physical Intelligence's π0.5 model, integrating the Multi-Scale Embodied Memory (MEM) architecture. This architecture enhances VLA (Vision-Language-Action) model performance in long-horizon robotic tasks through multi-scale memory mechanisms.

### Key Features

- **Short-term Visual Memory**: Fuses historical frame information via SpaceTimeSeparable attention
- **Long-term Language Memory**: Uses LLM to generate and manage semantic memory
- **Proprioceptive History**: Continuous state embeddings enhance action prediction
- **Backward Compatibility**: When MEM is disabled, behavior is identical to π0.5

---

## Architecture Design

### Memory Hierarchy

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
│                     Current Observation                    │
│              (image tokens + state tokens)                  │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Feature | Implementation |
|---------|----------------|
| SpaceTimeSeparable Attention | SpaceTimeSeparableBlock used every 4 layers |
| Causal Temporal Attention | Current frame can only attend to historical frames |
| History Token Dropping | Drop history tokens in upper layers to reduce computation |
| Language Memory Compression | LLM generates compressed memory summaries |

---

## Directory Structure

```
src/openpi/
├── models/
│   ├── siglip.py              ← SpaceTimeSeparableBlock + video encoder
│   ├── model.py               ← Observation with history frame fields
│   ├── pi0.py                 ← MEM forward pass logic
│   ├── pi0_config.py          ← MEMConfig configuration class
│   ├── high_level_policy.py   ← High-level policy (subtask + memory update)
│   └── memory_manager.py       ← Language memory manager
├── policies/
│   └── policy.py              ← MEMPolicy inference interface
├── training/
│   └── data_loader.py         ← MEMLeRobotDataset
├── transforms.py              ← VideoFrameStack + TokenizeMemory
└── scripts/
    └── gen_memory_labels.py   ← Memory label generation script
tests/
├── test_video_encoder.py
├── test_mem_model.py
├── test_memory_manager.py
├── test_transforms_mem.py
└── test_policy_mem.py
```

---

## New Files

| File | Description |
|------|-------------|
| `high_level_policy.py` | High-level policy: subtask scheduling and language memory update |
| `memory_manager.py` | Language memory manager: generation, update, compression, serialization |
| `gen_memory_labels.py` | Offline language memory label generation script |

## Modified Files

| File | Changes |
|------|---------|
| `siglip.py` | Added SpaceTimeSeparableBlock |
| `model.py` | Added image_history, tokenized_memory fields |
| `pi0.py` | Added MEM forward pass logic |
| `pi0_config.py` | Added MEMConfig configuration class |
| `policy.py` | Added MEMPolicy class |
| `transforms.py` | Added VideoFrameStack, TokenizeMemory |
| `data_loader.py` | Added MEMLeRobotDataset |

---

## Configuration

### MEMConfig

```python
@dataclass
class MEMConfig:
    # Short-term visual memory
    use_video_memory: bool = False          # Enable video memory
    video_memory_frames: int = 6            # Number of history frames K
    temporal_attn_every_n_layers: int = 4  # Insert temporal attention every n layers
    drop_history_tokens_after_layer: int = -4  # Drop history tokens after layer

    # Long-term language memory
    use_language_memory: bool = False       # Enable language memory
    max_memory_tokens: int = 256            # Maximum token count
    memory_loss_weight: float = 0.1          # Memory loss weight

    # Proprioceptive history
    use_state_history: bool = False          # Enable state history
    state_history_frames: int = 6            # Number of state history frames
```

---

## Constraints

| Constraint | Description |
|------------|-------------|
| **Backward Compatible** | When `image_history=None`, all MEM branches are skipped, behavior is identical to π0.5 |
| **Zero New Parameters** | Video encoder only modifies attention pattern, no new learnable parameters |
| **K=1 Numerical Equivalence** | With single frame, output is numerically identical to original SigLIP ViT (error < 1e-5) |
| **Real-time** | Inference latency < 300ms with 6 frames (H100 single GPU, 4 cameras) |

---

## Testing

Run tests:
```bash
# Video encoder tests
python -m pytest tests/test_video_encoder.py -v

# MEM model tests
python -m pytest tests/test_mem_model.py -v

# Memory manager tests
python -m pytest tests/test_memory_manager.py -v

# Inference policy tests
python -m pytest tests/test_policy_mem.py -v

# Data transform tests
python -m pytest tests/test_transforms_mem.py -v
```

---

## References

- Original Paper: [MEM: Multi-Scale Embodied Memory for Vision Language Action Models](https://pi.website/research/memory)
- Base Code: [Physical Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

---

## License

Inherited from the openpi project under Apache 2.0 License
