# Pi0.6 Training Dockerfile
# Base: CUDA 12.4 + cuDNN 9 (matches jax[cuda12]==0.5.3)
#
# 若出现: permission denied ... /.docker/buildx/activity/default
# 原因多为曾用 sudo docker 导致该文件属主为 root。任选其一:
#   sudo chown -R "$USER:$USER" ~/.docker/buildx/activity
#   DOCKER_BUILDKIT=0 docker build -t openpi-train -f Dockerfile .
#
# 使用 NGC (nvcr.io) 而非 docker.io，避免 registry.docker-cn.com 等镜像站超时/不转发 nvidia 官方库。
# 若仍失败：检查 /etc/docker/daemon.json 的 registry-mirrors，或暂时注释掉镜像配置后 sudo systemctl restart docker。
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:$PATH"

WORKDIR /workspace

# 源码与配置（datasets/ 见 .dockerignore，运行时挂载到 /workspace/datasets）
COPY . /workspace/

# Install Python dependencies via uv
RUN uv sync --no-dev --python python3.11

# Environment variables for training
ENV PYTHONPATH="/workspace/src:/workspace/packages/openpi-client/src"
ENV HF_LEROBOT_HOME="/workspace/datasets"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.92

# Checkpoint will be mounted or downloaded at runtime
# Default checkpoint cache location
ENV OPENPI_CHECKPOINT_DIR="/workspace/checkpoints/pretrained"

# 构建: docker build -t openpi-train -f Dockerfile .
# 训练（挂载数据集与 checkpoint 目录，需 NVIDIA Container Toolkit）:
# docker run --gpus all -it --rm \
#   -v /path/to/pi0.6/datasets:/workspace/datasets \
#   -v /path/to/checkpoints:/workspace/checkpoints \
#   openpi-train -c "bash train_g1_pick_apple.sh my_exp"

ENTRYPOINT ["/bin/bash"]
