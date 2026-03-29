# openpi (π0.6 自定义版本)

openpi 是 [Physical Intelligence 团队](https://www.physicalintelligence.company/) 发布的机器人开源模型和代码库。

本仓库是基于 π₀.₅ 和 π₀.₆ 论文的自定义实现，针对特定研究目的进行了修改。

目前，本仓库包含三种模型：
- [π₀ 模型](https://www.physicalintelligence.company/blog/pi0)，基于流的视觉-语言-动作模型 (VLA)
- [π₀-FAST 模型](https://www.physicalintelligence.company/research/fast)，基于 FAST 动作分词器的自回归 VLA
- [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05)，π₀ 的升级版本，通过[知识隔离训练](https://www.physicalintelligence.company/research/knowledge_insulation)实现了更好的开放世界泛化。注意，本仓库目前仅支持 π₀.₅ 的流匹配头进行训练和推理。

对于所有模型，我们提供预训练在 10k+ 小时机器人数据上的基础模型检查点，以及开箱即用或微调至用户自有数据集的示例。

这是一个实验：π₀ 是为我们自己的机器人开发的，这些机器人与广泛使用的平台（如 [ALOHA](https://tonyzhaozh.github.io/aloha/) 和 [DROID](https://droid-dataset.github.io/)）不同。尽管我们乐观地认为研究人员和实践者将能够进行创新性的新实验，将 π₀ 适配到他们自己的平台上，但我们并不期望每一种尝试都能成功。可以说：π₀ 可能适合或不适合您，但欢迎您尝试！

## 更新

- [2025年9月] 我们在 openpi 中添加了 PyTorch 支持。
- [2025年9月] 我们发布了 π₀.₅，这是 π₀ 的升级版本，具有更好的开放世界泛化能力。
- [2025年9月]：我们为 DROID 训练添加了[改进的空闲过滤器](examples/droid/README_train.md#data-filtering)。
- [2025年6月]：我们添加了使用 `openpi` 在完整 [DROID 数据集](https://droid-dataset.github.io/) 上训练 VLA 的[说明](examples/droid/README_train.md)。这是训练 pi0-FAST-DROID 所用训练流程的开源近似实现。

## 环境要求

要运行本仓库中的模型，您需要至少具有以下规格的 NVIDIA GPU。这些估算基于单个 GPU，但您也可以使用多个 GPU 并通过模型并行来减少每个 GPU 的内存需求，方法是在训练配置中配置 `fsdp_devices`。请注意，当前的训练脚本尚不支持多节点训练。

| 模式               | 所需内存 | 示例 GPU        |
| ------------------ | --------------- | ------------------ |
| 推理          | > 8 GB          | RTX 4090           |
| 微调 (LoRA) | > 22.5 GB       | RTX 4090           |
| 微调 (完整) | > 70 GB         | A100 (80GB) / H100 |

该仓库已在 Ubuntu 22.04 上测试，目前不支持其他操作系统。

## 安装

克隆本仓库时，请确保更新子模块：

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# 或者如果您已经克隆了仓库：
git submodule update --init --recursive
```

我们使用 [uv](https://docs.astral.sh/uv/) 来管理 Python 依赖。请参阅 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/) 进行设置。安装 uv 后，运行以下命令设置环境：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

注意：`GIT_LFS_SKIP_SMUDGE=1` 是拉取 LeRobot 作为依赖项所必需的。

**Docker**：作为 uv 安装的替代方案，我们提供使用 Docker 安装 openpi 的说明。如果您遇到系统设置问题，请考虑使用 Docker 来简化安装。请参阅 [Docker 设置](docs/docker.md) 了解更多详情。

## 模型检查点

### 基础模型

我们提供多个基础 VLA 模型检查点。这些检查点已预训练在 10k+ 小时的机器人数据上，可用于微调。

| 模型        | 使用场景    | 说明                                                                                                 | 检查点路径                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | 微调 | 基础 [π₀ 模型](https://www.physicalintelligence.company/blog/pi0) 用于微调                | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | 微调 | 基础自回归 [π₀-FAST 模型](https://www.physicalintelligence.company/research/fast) 用于微调 | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$    | 微调 | 基础 [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05) 用于微调    | `gs://openpi-assets/checkpoints/pi05_base`      |

### 微调模型

我们还为各种机器人平台和任务提供"专家"检查点。这些模型是从上述基础模型微调而来的，用于直接在目标机器人上运行。这些检查点可能适合或不适合您的特定机器人。由于这些检查点是在使用更广泛使用的机器人（如 ALOHA 和 DROID Franka 设置）收集的相对较小的数据集上进行微调的，它们可能无法泛化到您的特定设置，但我们发现其中一些（尤其是 DROID 检查点）在实践中泛化得相当广泛。

| 模型                    | 使用场景    | 说明                                                                                                                                                                                              | 检查点路径                                       |
| ------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| $\pi_0$-FAST-DROID       | 推理   | 在 [DROID 数据集](https://droid-dataset.github.io/) 上微调的 π₀-FAST 模型：可以在 DROID 机器人平台上对新场景执行各种简单的桌面操作任务 0-shot | `gs://openpi-assets/checkpoints/pi0_fast_droid`       |
| $\pi_0$-DROID            | 微调 | 在 [DROID 数据集](https://droid-dataset.github.io/) 上微调的 π₀ 模型：推理速度比 $\pi_0$-FAST-DROID 快，但语言指令遵循能力可能较差                                | `gs://openpi-assets/checkpoints/pi0_droid`            |
| $\pi_0$-ALOHA-towel      | 推理   | 在内部 [ALOHA](https://tonyzhaozh.github.io/aloha/) 数据上微调的 π₀ 模型：可以在 ALOHA 机器人平台上 0-shot 折叠各种毛巾                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_towel`      |
| $\pi_0$-ALOHA-tupperware | 推理   | 在内部 [ALOHA](https://tonyzhaozh.github.io/aloha/) 数据上微调的 π₀ 模型：可以从保鲜盒中取出食物                                                                                                             | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap  | 推理   | 在公开 [ALOHA](https://dit-policy.github.io/) 数据上微调的 π₀ 模型：可以打开笔帽                                                                                                          | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |
| $\pi_{0.5}$-LIBERO      | 推理   | 为 [LIBERO](https://libero-project.github.io/datasets) 基准微调的 π₀.₅ 模型：获得最先进性能（参见 [LIBERO README](examples/libero/README.md)) | `gs://openpi-assets/checkpoints/pi05_libero`      |
| $\pi_{0.5}$-DROID      | 推理 / 微调 | 在使用[知识隔离](https://www.physicalintelligence.company/research/knowledge_insulation) 的 [DROID 数据集](https://droid-dataset.github.io/) 上微调的 π₀.₅ 模型：推理快速且语言指令遵循能力好 | `gs://openpi-assets/checkpoints/pi05_droid`      |

默认情况下，检查点会自动从 `gs://openpi-assets` 下载，并在需要时缓存到 `~/.cache/openpi`。您可以通过设置 `OPENPI_DATA_HOME` 环境变量来覆盖下载路径。

## 运行预训练模型的推理

我们的预训练模型检查点可以用几行代码运行（这里以我们的 π₀-FAST-DROID 模型为例）：
```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# 创建训练好的策略
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 在虚拟示例上运行推理
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```

您也可以在[示例笔记本](examples/inference.ipynb)中测试。

我们提供在 [DROID](examples/droid/README.md) 和 [ALOHA](examples/aloha_real/README.md) 机器人上运行预训练检查点推理的详细分步示例。

**远程推理**：我们提供运行模型**远程**推理的[示例和代码](docs/remote_inference.md)：模型可以在不同的服务器上运行，并通过 websocket 连接将动作流式传输到机器人。这使得使用机器人外更强大的 GPU 变得容易，并保持机器人和策略环境分离。

**无需机器人测试推理**：我们提供一个[脚本](examples/simple_client/README.md)用于在没有机器人的情况下测试推理。该脚本将生成随机观测并使用模型运行推理。请参阅[此处](examples/simple_client/README.md)了解更多详情。

## 在您自己的数据上微调基础模型

我们将使用 [LIBERO 数据集](https://libero-project.github.io/datasets) 微调 π₀.₅ 模型作为示例，说明如何在自己的数据上微调基础模型。我们将解释三个步骤：
1. 将数据转换为 LeRobot 数据集（我们用于训练）
2. 定义训练配置并运行训练
3. 启动策略服务器并运行推理

### 1. 将数据转换为 LeRobot 数据集

我们提供了一个将 LIBERO 数据转换为 LeRobot 数据集的最小示例脚本，位于 [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py)。您可以轻松修改它来转换您自己的数据！您可以从[这里](https://huggingface.co/datasets/openvla/modified_libero_rlds)下载原始 LIBERO 数据集，并使用以下命令运行脚本：

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**注意**：如果您只想微调 LIBERO，您可以跳过此步骤，因为我们的 LIBERO 微调配置指向预转换的 LIBERO 数据集。此步骤只是一个您可以适配到自己的数据的示例。

### 2. 定义训练配置并运行训练

要在自己的数据上微调基础模型，您需要定义数据处理和训练的配置。我们提供如下所示的带详细注释的 LIBERO 示例配置，您可以为自己的数据集进行修改：

- [`LiberoInputs` 和 `LiberoOutputs`](src/openpi/policies/libero_policy.py)：定义从 LIBERO 环境到模型的数据映射，反之亦然。训练和推理都会使用。
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py)：定义如何从 LeRobot 数据集处理原始 LIBERO 数据用于训练。
- [`TrainConfig`](src/openpi/training/config.py)：定义微调超参数、数据配置和权重加载器。

我们为 LIBERO 数据上的 [π₀](src/openpi/training/config.py)、[π₀-FAST](src/openpi/training/config.py) 和 [π₀.₅](src/openpi/training/config.py) 提供示例微调配置。

��运行训练之前，我们需要计算训练数据的归一化统计量。使用您的训练配置名称运行以下脚本：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

现在我们可以使用以下命令启动训练（`--overwrite` 标志用于在使用相同配置重新运行微调时覆盖现有检查点）：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

该命令会将训练进度记录到控制台并保存检查点到 `checkpoints` 目录。您还可以在 Weights & Biases 仪表板上监控训练进度。为了最大限度地使用 GPU 内存，请在运行训练之前设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`——这使 JAX 能够使用高达 90% 的 GPU 内存（默认 75%）。

**注意**：我们提供从预训练中*重新加载*状态/动作归一化统计量的功能。如果您正在微调我们预训练混合物中包含的机器人的新任务，这可能会有益处。有关如何重新加载归一化统计量的更多详细信息，请参阅 [norm_stats.md](docs/norm_stats.md) 文件。

### 3. 启动策略服务器并运行推理

训练完成后，我们可以通过启动策略服务器然后从 LIBERO 评估脚本查询它来运行推理。启动模型服务器很容易（本例使用迭代 20,000 的检查点，根据需要修改）：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

这将启动一个监听端口 8000 的服务器，并等待向它发送观测值。然后我们可以运行评估脚本（或机器人运行时）来查询服务器。

特别是对于运行 LIBERO 评估，我们提供（并推荐使用）一个 Docker 化的工作流程，同时处理策略服务器和评估脚本。请参阅 [LIBERO README](examples/libero/README.md) 了解更多详情。

如果您想将策略服务器调用嵌入到您自己的机器人运行时，我们有一个如何在[远程推理文档](docs/remote_inference.md)中执行此操作的最小示例。

## 更多示例

我们提供了更多关于如何在 ALOHA 平台上微调和运行模型推理的示例，请参阅以下 README：
- [ALOHA 模拟器](examples/aloha_sim)
- [ALOHA 真实机器人](examples/aloha_real)
- [UR5](examples/ur5)

## PyTorch 支持

openpi 现在提供 π₀ 和 π₀.₅ 模型的 PyTorch 实现，与原始 JAX 版本一起！PyTorch 实现已在 LIBERO 基准上验证（推理和微调）。目前不支持一些功能（未来可能会改变）：

- π₀-FAST 模型
- 混合精度训练
- FSDP（全分片数据并行）训练
- LoRA（低秩适配）训练
- 训练期间的 EMA（指数移动平均）权重

### 设置
1. 确保您已安装所有依赖项的最新版本：`uv sync`

2. 仔细检查您已安装 transformers 4.53.2：`uv pip show transformers`

3. 应用 transformers 库补丁：
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```

这将覆盖 transformers 库中的几个文件以实现必要的模型更改：1) 支持 AdaRMS，2) 正确控制激活精度，3) 允许使用 KV 缓存而不更新它。

**警告**：使用默认的 uv 链接模式（硬链接），这将永久影响 uv 缓存中的 transformers 库，意味着更改将保留在重新安装 transformers 期间，甚至可能传播到使用 transformers 的其他项目。要完全撤消此操作，您必须运行 `uv cache clean transformers`。

### 将 JAX 模型转换为 PyTorch

将 JAX 模型检查点转换为 PyTorch 格式：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### 使用 PyTorch 运行推理

PyTorch 实现使用与 JAX 版本相同的 API——您只需要更改检查点路径以指向转换后的 PyTorch 模型：

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# 创建训练好的策略（自动检测 PyTorch 格式）
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 运行推理（与 JAX 相同的 API）
action_chunk = policy.infer(example)["actions"]
```

### PyTorch 策略服务器

策略服务器与 PyTorch 模型的工作方式完全相同——只需指向转换后的检查点目录：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### 使用 PyTorch 微调

要在 PyTorch 中微调模型：

1. 将 JAX 基础模型转换为 PyTorch 格式：
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. 在配置中使用 `pytorch_weight_path` 指定转换后的 PyTorch 模型路径

3. 使用以下模式之一启动训练：

```bash
# 单 GPU 训练：
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# 示例：
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # 从最新检查点恢复

# 多 GPU 训练（单节点）：
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# 示例：
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# 多节点训练：
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### 精度设置

JAX 和 PyTorch 实现按以下方式处理精度：

**JAX：**
1. 推理：大多数权重和计算使用 bfloat16，少数计算使用 float32 以保持稳定性
2. 训练：默认使用混合精度：权重和梯度使用 float32，（大多数）激活和计算使用 bfloat16。您可以通过在配置中将 `dtype` 设置为 float32 来更改为完整的 float32 训练。

**PyTorch：**
1. 推理：与 JAX 匹配——大多数权重和计算使用 bfloat16，少数权重转换为 float32 以保持稳定性
2. 训练：支持完整的 bfloat16（默认）或完整的 float32。您可以通过在配置中设置 `pytorch_training_precision` 来更改它。bfloat16 使用更少的内存，但与 float32 相比显示更高的损失。不支持混合精度。

使用 torch.compile，JAX 和 PyTorch 之间的推理速度相当。

## 故障排除

我们将在此处收集常见问题及其解决方案。如果您遇到问题，请首先检查此处。如果您找不到解决方案，请在仓库上提交问题（请参阅[此处](CONTRIBUTING.md)了解指南）。

| 问题                                     | 解决                                                                                                                                                                                   |
| ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `uv sync` 因依赖冲突失败 | 尝试删除虚拟环境目录（`rm -rf .venv`）并再次运行 `uv sync`。如果问题仍然存在，请确保您已安装最新版本的 `uv`（`uv self update`）。 |
| 训练耗尽 GPU 内存           | 在运行训练之前确保设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`（或更高），以允许 JAX 使用更多 GPU 内存。您也可以使用 `--fsdp-devices <n>`，其中 `<n>` 是您的 GPU 数量，以启用[全分片数据并行](https://engineering.fb.com/2021/07/15/open-source/fsdp/)，这会以较慢的训练换取更少的内存使用（减速程度取决于您的具体设置）。如果仍然内存不足，您可能需要考虑禁用 EMA。        |
| 策略服务器连接错误           | 检查服务器是否正在运行并监听预期端口。验证客户端和服务器之间的网络连接和防火墙设置。                                            |
| 训练时缺少归一化统计量错误 | 在开始训练之前，使用您的配置名称运行 `scripts/compute_norm_stats.py`。                                                                                                          |
| 数据集下载失败                    | 检查您的互联网连接。对于 HuggingFace 数据集，确保您已登录（`huggingface-cli login`）。                                                                                 |
| CUDA/GPU 错误                           | 验证 NVIDIA 驱动已正确安装。对于 Docker，确保已安装 nvidia-container-toolkit。检查 GPU 兼容性。您无需在系统级别安装 CUDA 库——它们将通过 uv 安装。您甚至可能想要*卸载*系统 CUDA 库（如果遇到 CUDA 问题），因为系统库有时可能会导致冲突。 |
| 运行示例时导入错误 | 确保已使用 `uv sync` 安装所有依赖项。一些示例的 README 中可能列有额外的要求。                    |
| 动作维度不匹配                | 验证您的数据处理转换与机器人的预期输入/输出维度匹配。检查策略类中的动作空间定义。                                  |
| 训练损���发���                            | 检查数据集的 `norm_stats.json` 中的 `q01`、`q99` 和 `std` 值。某些很少使用的维度的 `q01`、`q99` 或 `std` 值可能会非常小，导致归一化后产生非常大的状态和动作。您可以手动调整归一化统计量作为解决方法。 |

---

# π0.6-MEM：用于视觉-语言-动作模型的多尺度具身记忆

> 基于 openpi (π0.5) 代码库实现的多尺度具身记忆 (MEM) 架构
> 参考论文：[MEM: Multi-Scale Embodied Memory for Vision Language Action Models](https://pi.website/research/memory)

---

## 项目概述

π0.6-MEM 是 Physical Intelligence π0.5 模型的扩展版本，集成了多尺度具身记忆 (MEM) 架构。该架构通过多尺度记忆机制增强了 VLA (视觉-语言-动作) 模型在长时序机器人任务中的性能。

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
│                    长期语言记忆                             │
│                 (LLM 生成的语义记忆)                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              短期视觉记忆 (K 帧)                           │
│         时空分离块 + 时间注意力                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     当前观测                                │
│              (图像 tokens + 状态 tokens)                    │
└─────────────────────────────────────────────────────────────┘
```

### 关键设计

| 特性 | 实现 |
|------|------|
| 时空分离注意力 | 每 4 层使用一次时空分离块 |
| 因果时间注意力 | 当前帧只能 attend 到历史帧 |
| 历史帧 token 丢弃 | 在上层丢弃，减少计算量 |
| 语言记忆压缩 | LLM 生成压缩后的记忆摘要 |

---

## 目录结构

```
src/openpi/
├── models/
│   ├── siglip.py              ← 时空分离块 + 视频编码器
│   ├── model.py               ← Observation 新增历史帧字段
│   ├── pi0.py                 ← MEM 前向传播逻辑
│   ├── pi0_config.py          ← MEMConfig 配置类
│   ├── high_level_policy.py   ← 高层策略（子任务 + 记忆更新）
│   └── memory_manager.py       ← 语言记忆管理
├── policies/
│   └── policy.py              ← MEMPolicy 推理接口
├── training/
│   └── data_loader.py         ← MEMLeRobotDataset
├── transforms.py              ← 视频帧堆叠 + 记忆分词
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
| `siglip.py` | 添加时空分离块 |
| `model.py` | 添加历史图像、记忆分词等字段 |
| `pi0.py` | 添加 MEM 前向传播逻辑 |
| `pi0_config.py` | 添加 MEMConfig 配置类 |
| `policy.py` | 添加 MEMPolicy 类 |
| `transforms.py` | 添加视频帧堆叠、记忆分词 |
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

## 许可证

继承自 openpi 项目的 Apache 2.0 许可证