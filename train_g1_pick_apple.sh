#!/bin/bash
# Training script for G1 pick-apple dataset
# Dataset: PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-apple
# Model: pi05 with 43-dim action space
#
# Native install requires glibc >= 2.31 (PyTorch / lerobot / rerun-sdk wheels).
# On older hosts, use Docker: see Dockerfile and README.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

GLIBC_VER="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "0")"
GLIBC_MINOR="${GLIBC_VER#*.}"
GLIBC_MAJOR="${GLIBC_VER%%.*}"
# 容器内（含 openpi 预置镜像）不检查宿主机 glibc
if [ ! -f /.dockerenv ]; then
  if [ -z "$GLIBC_VER" ] || [ "$GLIBC_VER" = "0" ]; then
    :
  elif [ "$GLIBC_MAJOR" -lt 2 ] 2>/dev/null || { [ "$GLIBC_MAJOR" -eq 2 ] 2>/dev/null && [ "${GLIBC_MINOR:-0}" -lt 31 ] 2>/dev/null; }; then
    echo "错误: 检测到 glibc ${GLIBC_VER}。本仓库在宿主机上通过 uv 安装时，依赖需要 glibc >= 2.31 的预编译包。"
    echo "请在 glibc 较新的机器上训练，或使用: docker build -f Dockerfile .  在镜像内执行本脚本。"
    exit 1
  fi
fi

DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/datasets}"

export PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR/packages/openpi-client/src"
export HF_LEROBOT_HOME="$DATASET_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92

cd "$SCRIPT_DIR"

# openpi:latest 等镜像：依赖装在 /.venv，代码通过 -v 挂载到 /workspace（.pth 指向 /workspace/src）
if [ -f /.dockerenv ] && [ -x "/.venv/bin/python" ]; then
  PYTHON="/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  PYTHON="$(cd "$SCRIPT_DIR" && uv run python -c "import sys; print(sys.executable)")"
else
  PYTHON="${PYTHON:-python3}"
fi

# Step 1: Compute normalization stats (if not already done)
NORM_STATS="$SCRIPT_DIR/assets/pi05_g1_pick_apple/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-apple/norm_stats.json"
if [ ! -f "$NORM_STATS" ]; then
    echo "Computing normalization stats..."
    $PYTHON scripts/compute_norm_stats.py --config-name pi05_g1_pick_apple
else
    echo "Norm stats already exist: $NORM_STATS"
fi

# Step 2: Run training
# Use pi05_g1_pick_apple_lora for RTX 3090 (24GB), pi05_g1_pick_apple for A100/H100 (>70GB)
CONFIG="${CONFIG:-pi05_g1_pick_apple_lora}"
EXP_NAME="${1:-g1_pick_apple_run1}"
CKPT_DIR="$SCRIPT_DIR/checkpoints/$CONFIG/$EXP_NAME"

TRAIN_EXTRA=(--no-wandb-enabled)
if [ "${OVERWRITE:-0}" = "1" ]; then
  TRAIN_EXTRA+=(--overwrite)
elif [ "${RESUME:-0}" = "1" ]; then
  TRAIN_EXTRA+=(--resume)
elif [ -d "$CKPT_DIR" ] && [ -n "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then
  echo "检测到已有 checkpoint 目录，使用 --resume 续训。若要清空重训请设置 OVERWRITE=1"
  TRAIN_EXTRA+=(--resume)
else
  TRAIN_EXTRA+=(--overwrite)
fi

echo "Starting training: config=$CONFIG exp_name=$EXP_NAME ckpt=$CKPT_DIR"
$PYTHON scripts/train.py "$CONFIG" --exp-name="$EXP_NAME" "${TRAIN_EXTRA[@]}"

echo "Training complete! Checkpoint saved to:"
echo "  $CKPT_DIR/"
