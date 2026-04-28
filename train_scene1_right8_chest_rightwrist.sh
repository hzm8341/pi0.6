#!/bin/bash
# Training script for the local Scene 1 right-arm dataset.
# Dataset: datasets/scene1_right8_chest_rightwrist
# Model: pi05 LoRA fine-tuning with an 8-dim action space
#
# This script keeps the pretrained model mostly frozen and is the safest
# default for small custom datasets. If you later want a full fine-tune on
# an A100 80G box, start from the same config and remove the LoRA freeze.
#
# To override the environment:
#   VENV=/media/hzm/SSD_2T/GitHub/openpi/.venv
#   DATASET_ROOT=/data/datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

GLIBC_VER="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "0")"
GLIBC_MINOR="${GLIBC_VER#*.}"
GLIBC_MAJOR="${GLIBC_VER%%.*}"
if [ ! -f /.dockerenv ]; then
  if [ -z "$GLIBC_VER" ] || [ "$GLIBC_VER" = "0" ]; then
    :
  elif [ "$GLIBC_MAJOR" -lt 2 ] 2>/dev/null || { [ "$GLIBC_MAJOR" -eq 2 ] 2>/dev/null && [ "${GLIBC_MINOR:-0}" -lt 31 ] 2>/dev/null; }; then
    echo "错误: 检测到 glibc ${GLIBC_VER}。本仓库在宿主机上通过 uv 安装时，依赖需要 glibc >= 2.31 的预编译包。"
    echo "请在 glibc 较新的机器上训练，或使用: docker build -f Dockerfile .  在镜像内执行本脚本。"
    exit 1
  fi
fi

VENV="${VENV:-/media/hzm/SSD_2T/GitHub/openpi/.venv}"
DATASET_ROOT="${DATASET_ROOT:-/data/datasets}"

export PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR/packages/openpi-client/src"
export HF_LEROBOT_HOME="$DATASET_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92

cd "$SCRIPT_DIR"

if [ -x "$VENV/bin/python" ]; then
  PYTHON="$VENV/bin/python"
elif [ -f /.dockerenv ] && [ -x "/.venv/bin/python" ]; then
  PYTHON="/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  PYTHON="$(cd "$SCRIPT_DIR" && uv run python -c "import sys; print(sys.executable)")"
else
  PYTHON="${PYTHON:-python3}"
fi

NORM_STATS="$SCRIPT_DIR/assets/pi05_scene1_right8_chest_wrist_finetune/scene1_right8_chest_rightwrist/norm_stats.json"
if [ ! -f "$NORM_STATS" ]; then
  echo "Computing normalization stats..."
  $PYTHON scripts/compute_norm_stats.py --config-name pi05_scene1_right8_chest_rightwrist_lora
else
  echo "Norm stats already exist: $NORM_STATS"
fi

CONFIG="${CONFIG:-pi05_scene1_right8_chest_rightwrist_lora}"
EXP_NAME="${1:-scene1_right8_chest_rightwrist_run1}"

TRAIN_EXTRA=(--no-wandb-enabled)
if [ "${OVERWRITE:-0}" = "1" ]; then
  TRAIN_EXTRA+=(--overwrite)
elif [ "${RESUME:-0}" = "1" ]; then
  TRAIN_EXTRA+=(--resume)
else
  TRAIN_EXTRA+=(--overwrite)
fi

echo "Starting training: config=$CONFIG exp_name=$EXP_NAME"
$PYTHON scripts/train.py "$CONFIG" --exp-name="$EXP_NAME" "${TRAIN_EXTRA[@]}"

echo "Training complete."
