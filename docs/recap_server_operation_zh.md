# RECAP 服务器操作手册

这份说明用于你在 GPU 服务器上从“只有 LeRobot 数据集”开始，跑通 RECAP 离线标签生成、10 step 烟测训练和正式 `pi05_recap` 训练。

当前仓库实现的是离线版 RECAP 训练准备流程：把 LeRobot 数据转换成 RECAP episode JSON，生成 `advantage_indicator`、`use_advantage`、`is_human_intervention` 侧车字段，再用 `pi05_recap` 训练。它不是完整论文级 `pi0.6` 闭环实现，真实 rollout、完整 value model、人类接管 callback 仍在 TODO。

## 0. 你需要准备什么

服务器上至少需要：

- 本仓库代码，分支为 `codex/offline-recap-labeling`，或已经合并该分支的主分支。
- 一个本地 LeRobot 数据集目录，例如 `/data/lerobot/test_tube_dataset`。
- 训练 checkpoint 访问权限。若使用 GCS checkpoint，需要服务器能访问对应资源。
- Python/uv 环境和 GPU 驱动可用。
- 推荐先保留 100GB 以上空闲磁盘空间，用于 `outputs/`、缓存和 checkpoint。

LeRobot 数据集目录至少应包含：

```text
/data/lerobot/test_tube_dataset/
  meta/
    info.json
    episodes.jsonl
    tasks.jsonl
  data/
    chunk-000/
      episode_000000.parquet
      ...
  videos/
    chunk-000/
      observation.images.xxx/
        episode_000000.mp4
        ...
```

## 1. 拉取代码

```bash
cd /path/to/pi0.6
git fetch origin
git checkout codex/offline-recap-labeling
git pull
git status -sb
```

确认至少能看到这些文件：

```bash
ls scripts/convert_lerobot_to_recap_episodes.py
ls scripts/run_recap_gpu_validation.sh
ls src/openpi/training/lerobot_to_recap.py
```

## 2. 安装环境

```bash
cd /path/to/pi0.6
uv sync
```

如果服务器不能访问外网，需要提前配置 Python 包缓存或使用已有环境。安装后先确认脚本能启动：

```bash
PYTHONPATH=src uv run python scripts/convert_lerobot_to_recap_episodes.py --help
PYTHONPATH=src uv run python scripts/label_recap_advantage.py --help
```

如果使用 Hugging Face 数据或模型：

```bash
hf auth login
```

如果使用 W&B：

```bash
wandb login
```

## 3. 设置路径变量

下面示例假设：

- 仓库路径：`/path/to/pi0.6`
- LeRobot 数据集路径：`/data/lerobot/test_tube_dataset`
- 输出目录：仓库内 `outputs/`

```bash
cd /path/to/pi0.6

export LEROBOT_ROOT=/data/lerobot/test_tube_dataset
export RECAP_EPISODES=outputs/recap_episodes
export RECAP_LABELS=outputs/recap_labels
export RECAP_TRIMMED=outputs/recap_trimmed_episodes
export CONFIG_NAME=pi05_recap
export EXP_NAME=test_tube_recap_v1
```

检查 LeRobot 数据集：

```bash
test -f "${LEROBOT_ROOT}/meta/info.json"
test -f "${LEROBOT_ROOT}/meta/episodes.jsonl"
test -d "${LEROBOT_ROOT}/data"
```

## 4. 先做 1 到 2 个 episode 的转换烟测

```bash
PYTHONPATH=src uv run python scripts/convert_lerobot_to_recap_episodes.py \
  --lerobot-root "${LEROBOT_ROOT}" \
  --output-episodes /tmp/recap_episode_smoke \
  --default-success unknown \
  --max-episodes 2
```

预期输出类似：

```text
Wrote 2 RECAP episode(s) to /tmp/recap_episode_smoke
2 episode(s) have unknown success labels and need manual review.
```

检查 JSON 是否生成：

```bash
ls /tmp/recap_episode_smoke | head
PYTHONPATH=src uv run python scripts/eval_recap_episodes.py \
  --input-episodes /tmp/recap_episode_smoke \
  --output /tmp/recap_episode_smoke_summary.json
```

## 5. 转换完整 LeRobot 数据集

如果你暂时还没有 success/failure 标签，先用 `unknown`：

```bash
PYTHONPATH=src uv run python scripts/convert_lerobot_to_recap_episodes.py \
  --lerobot-root "${LEROBOT_ROOT}" \
  --output-episodes "${RECAP_EPISODES}" \
  --default-success unknown
```

注意：`--default-success unknown` 会把 JSON 里的 `success` 临时写成 `false`，同时设置：

```json
"metadata": {
  "success_needs_review": true
}
```

这只是为了让后续脚本可以读取，不代表这些 episode 都失败。正式训练前应该人工复核 success/failure。

## 6. 人工标注 success/failure

推荐创建一个 JSONL 文件，例如 `outputs/success_labels.jsonl`：

```json
{"episode_index": 0, "success": true}
{"episode_index": 1, "success": false}
{"episode_index": 2, "success": true}
```

也可以用 `episode_id`：

```json
{"episode_id": "episode_000000", "success": true}
{"episode_id": "episode_000001", "success": false}
```

有标签后重新转换：

```bash
PYTHONPATH=src uv run python scripts/convert_lerobot_to_recap_episodes.py \
  --lerobot-root "${LEROBOT_ROOT}" \
  --output-episodes "${RECAP_EPISODES}" \
  --success-labels outputs/success_labels.jsonl
```

如果你的数据集全部都是成功示教，也可以先用：

```bash
PYTHONPATH=src uv run python scripts/convert_lerobot_to_recap_episodes.py \
  --lerobot-root "${LEROBOT_ROOT}" \
  --output-episodes "${RECAP_EPISODES}" \
  --default-success true
```

但这会让 RECAP 的 success/failure reward 区分变弱。更好的做法是至少标出失败 episode。

## 7. 生成 RECAP 标签和训练侧车字段

完整流程可以用一键脚本：

```bash
EPISODES_DIR="${RECAP_EPISODES}" \
CONFIG_NAME="${CONFIG_NAME}" \
EXP_NAME="${EXP_NAME}" \
WANDB_ENABLED=False \
bash scripts/run_recap_gpu_validation.sh
```

这个脚本会依次做：

1. 生成 train/eval split manifest：`outputs/recap_labels/split_manifest.json`
2. 裁剪尾部静止帧：`outputs/recap_trimmed_episodes`
3. 生成离线摘要：`outputs/recap_labels/offline_eval_summary.json`
4. 生成每帧 reward/return/value/advantage：`outputs/recap_labels/recap_labels.jsonl`
5. 生成 LeRobot dataloader 侧车字段：`outputs/recap_labels/lerobot_fields.npz`
6. 检查侧车字段 dtype 和长度
7. 跑 RECAP 相关单元测试
8. 计算 norm stats
9. 跑 10 step `pi05_recap` 烟测训练

如果你想分步执行，命令如下：

```bash
PYTHONPATH=src uv run python scripts/create_recap_split_manifest.py \
  --input-episodes "${RECAP_EPISODES}" \
  --output "${RECAP_LABELS}/split_manifest.json" \
  --eval-fraction 0.15 \
  --seed 0

PYTHONPATH=src uv run python scripts/trim_recap_episodes.py \
  --input-episodes "${RECAP_EPISODES}" \
  --output-episodes "${RECAP_TRIMMED}" \
  --action-norm-threshold 0.05 \
  --min-frames 10

PYTHONPATH=src uv run python scripts/eval_recap_episodes.py \
  --input-episodes "${RECAP_TRIMMED}" \
  --output "${RECAP_LABELS}/offline_eval_summary.json" \
  --static-action-threshold 0.05

PYTHONPATH=src uv run python scripts/label_recap_advantage.py \
  --input-episodes "${RECAP_TRIMMED}" \
  --output-dir "${RECAP_LABELS}" \
  --positive-fraction 0.4 \
  --n-step-lookahead 50
```

关键输出是：

```text
outputs/recap_labels/lerobot_fields.npz
```

训练时会通过 `--data.recap-fields-path` 把它合并到 dataloader sample 中。

## 8. 检查 sidecar 长度

侧车字段长度必须和训练 dataloader 读到的 LeRobot flattened frame 数一致。先看 npz 内部长度：

```bash
PYTHONPATH=src uv run python - <<'PY'
import numpy as np
path = "outputs/recap_labels/lerobot_fields.npz"
data = np.load(path)
for key in data.files:
    print(key, data[key].shape, data[key].dtype)
PY
```

你应该看到：

```text
advantage_indicator (N,) int32 或 int64
use_advantage (N,) bool
is_human_intervention (N,) bool
```

如果训练时报长度不匹配，通常原因是：

- 你转换/裁剪的是一份数据，但训练 config 读的是另一份 LeRobot 数据。
- 你裁剪了 episode，但 dataloader 仍按未裁剪的 LeRobot 原始帧数读取。
- episode 顺序或 frame 顺序和 LeRobot flattened 顺序不一致。

当前实现是 sidecar 合并，不是真正原地写回 LeRobot parquet。因此必须保证生成 sidecar 的 episode 顺序与训练读取的 LeRobot 顺序一致。

## 9. 跑 10 step 烟测训练

如果第 7 步的一键脚本已经完成，这一步已经跑过。你也可以手动跑：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
PYTHONPATH=src uv run scripts/train.py "${CONFIG_NAME}" \
  --data.recap-fields-path "${RECAP_LABELS}/lerobot_fields.npz" \
  --exp-name "${EXP_NAME}_smoke" \
  --overwrite
```

如果有 max steps 参数，以本仓库当前脚本支持的 tyro 参数为准；不确定时先运行：

```bash
PYTHONPATH=src uv run scripts/train.py "${CONFIG_NAME}" --help
```

烟测目标不是收敛，而是确认：

- 数据能读到。
- norm stats 能算。
- `lerobot_fields.npz` 能合并。
- loss 不 NaN。
- GPU 显存不会立刻 OOM。

## 10. 正式训练

烟测通过后，启动正式训练：

```bash
EPISODES_DIR="${RECAP_EPISODES}" \
CONFIG_NAME="${CONFIG_NAME}" \
EXP_NAME="${EXP_NAME}" \
WANDB_ENABLED=True \
RUN_FULL_TRAIN=1 \
bash scripts/run_recap_gpu_validation.sh
```

或者直接训练：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
PYTHONPATH=src uv run scripts/train.py "${CONFIG_NAME}" \
  --data.recap-fields-path "${RECAP_LABELS}/lerobot_fields.npz" \
  --exp-name "${EXP_NAME}" \
  --overwrite
```

建议正式训练前记录：

```bash
git rev-parse HEAD
nvidia-smi
du -sh "${LEROBOT_ROOT}" outputs
cat "${RECAP_LABELS}/offline_eval_summary.json"
```

## 11. 训练完先做离线检查

不要训练完直接上真机插试管。先检查：

- 训练 loss 是否出现 NaN 或突然爆炸。
- action 范围是否明显超出原始数据。
- gripper 通道是否合理。
- eval split 上的 loss 是否明显劣化。
- 成功/失败 episode 的 advantage 分布是否符合预期。

可以查看：

```bash
ls outputs/recap_labels
cat outputs/recap_labels/offline_eval_summary.json
```

## 12. 常见错误

### 找不到 `lerobot_to_recap.py`

说明服务器代码不是最新分支：

```bash
git fetch origin
git checkout codex/offline-recap-labeling
git pull
```

### 找不到 parquet

检查 `meta/info.json` 里的 `data_path` 是否和真实目录一致：

```bash
cat "${LEROBOT_ROOT}/meta/info.json"
find "${LEROBOT_ROOT}/data" -name '*.parquet' | head
```

### success 全部 unknown

这是正常的初始状态，但不建议直接正式训练。创建 `outputs/success_labels.jsonl` 后重新转换。

### `RECAP field ... must have the same length as the dataset`

这是 sidecar 长度和 LeRobot dataloader 长度不一致。优先确认：

```bash
PYTHONPATH=src uv run python - <<'PY'
import numpy as np
data = np.load("outputs/recap_labels/lerobot_fields.npz")
print({k: v.shape for k, v in data.items()})
PY
```

然后确认训练 config 读取的是同一份 LeRobot 数据集。

### CUDA/JAX OOM

处理顺序：

1. 降 batch size。
2. 设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` 或 `0.8`。
3. 关闭其他占 GPU 的进程。
4. 确认没有同时跑多个训练任务。

### loss NaN

优先检查：

- action 是否有 NaN/Inf。
- norm stats 是否由同一份数据生成。
- gripper/action 维度是否和 config 匹配。
- success 标签是否全错。

## 13. 最短命令清单

如果只想按顺序复制执行，把路径改成你的服务器路径：

```bash
cd /path/to/pi0.6
git fetch origin
git checkout codex/offline-recap-labeling
git pull
uv sync

export LEROBOT_ROOT=/data/lerobot/test_tube_dataset
export RECAP_EPISODES=outputs/recap_episodes
export CONFIG_NAME=pi05_recap
export EXP_NAME=test_tube_recap_v1

PYTHONPATH=src uv run python scripts/convert_lerobot_to_recap_episodes.py \
  --lerobot-root "${LEROBOT_ROOT}" \
  --output-episodes "${RECAP_EPISODES}" \
  --default-success unknown

# 建议在这里补 outputs/success_labels.jsonl 后重新转换。

EPISODES_DIR="${RECAP_EPISODES}" \
CONFIG_NAME="${CONFIG_NAME}" \
EXP_NAME="${EXP_NAME}" \
WANDB_ENABLED=False \
bash scripts/run_recap_gpu_validation.sh

EPISODES_DIR="${RECAP_EPISODES}" \
CONFIG_NAME="${CONFIG_NAME}" \
EXP_NAME="${EXP_NAME}" \
WANDB_ENABLED=True \
RUN_FULL_TRAIN=1 \
bash scripts/run_recap_gpu_validation.sh
```
