#!/bin/bash
# 使用本机已有的 openpi 镜像（如 openpi:latest / 7cf4590191db）启动 G1 训练。
# 仓库与 checkpoints 挂载到容器内 /workspace；镜像内需有 /.venv（Python 3.11）。
#
# 用法:
#   OPENPI_DOCKER_IMAGE=openpi:latest bash run_train_openpi_image.sh [传给 train_g1_pick_apple.sh 的参数]
# 例:
#   OPENPI_DOCKER_IMAGE=7cf4590191db bash run_train_openpi_image.sh g1_run1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${OPENPI_DOCKER_IMAGE:-openpi:latest}"

docker run --gpus all --rm -it \
  -v "$SCRIPT_DIR:/workspace" \
  -v "$SCRIPT_DIR/checkpoints:/workspace/checkpoints" \
  -e HF_LEROBOT_HOME="/workspace/datasets" \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
  -w /workspace \
  "$IMAGE" \
  bash -lc 'bash train_g1_pick_apple.sh '"$(printf '%q ' "$@")"
