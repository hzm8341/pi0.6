#!/bin/bash
# Environment setup script for pi0.6 training on remote server
# Usage: bash setup_env.sh [--skip-checkpoint]
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_CHECKPOINT=false
for arg in "$@"; do
    [[ "$arg" == "--skip-checkpoint" ]] && SKIP_CHECKPOINT=true
done

echo "====== Pi0.6 Environment Setup ======"

GLIBC_VER="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "unknown")"
GLIBC_MINOR="${GLIBC_VER#*.}"
GLIBC_MAJOR="${GLIBC_VER%%.*}"
if [ "$GLIBC_VER" != "unknown" ] && [ -n "$GLIBC_VER" ]; then
  if [ "$GLIBC_MAJOR" -lt 2 ] 2>/dev/null || { [ "$GLIBC_MAJOR" -eq 2 ] 2>/dev/null && [ "${GLIBC_MINOR:-0}" -lt 31 ] 2>/dev/null; }; then
    echo "警告: 当前 glibc 为 ${GLIBC_VER}，低于 2.31，uv sync 很可能因 PyTorch/rerun-sdk 等轮子无法安装而失败。"
    echo "请使用本仓库 Dockerfile 在 Ubuntu 22.04 / CUDA 基础镜像中安装，或换用较新系统。"
    echo ""
  fi
fi

# ---- 1. Install uv ----
if ! command -v uv &>/dev/null; then
    echo "[1/4] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"' >> ~/.bashrc
else
    echo "[1/4] uv already installed: $(uv --version)"
fi

# ---- 2. Install Python dependencies ----
echo "[2/4] Installing Python dependencies..."
cd "$SCRIPT_DIR"
uv sync --no-dev --python python3.11
echo "Dependencies installed."

# ---- 3. System packages check ----
echo "[3/4] Checking system packages..."
MISSING=()
for pkg in ffmpeg libgl1 git; do
    dpkg -s "$pkg" &>/dev/null 2>&1 || MISSING+=("$pkg")
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "Installing missing system packages: ${MISSING[*]}"
    sudo apt-get update -qq && sudo apt-get install -y "${MISSING[@]}"
fi
echo "System packages OK."

# ---- 4. Download pi05_base checkpoint ----
CHECKPOINT_DIR="$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base"
if $SKIP_CHECKPOINT; then
    echo "[4/4] Skipping checkpoint download (--skip-checkpoint)."
elif [ -f "$CHECKPOINT_DIR/params/commit_success.txt" ]; then
    echo "[4/4] Checkpoint already exists at $CHECKPOINT_DIR"
else
    echo "[4/4] Downloading pi05_base checkpoint (~15GB)..."
    mkdir -p "$CHECKPOINT_DIR"
    PYTHON="$(cd "$SCRIPT_DIR" && uv run which python)"
    export PYTHONPATH="$SCRIPT_DIR/src:$SCRIPT_DIR/packages/openpi-client/src"
    # Use openpi's built-in downloader
    $PYTHON -c "
import openpi.training.weight_loaders as wl
loader = wl.CheckpointWeightLoader('gs://openpi-assets/checkpoints/pi05_base/params')
print('Checkpoint download triggered via weight loader on first training run.')
"
    echo "Note: Checkpoint will be auto-downloaded on first training run."
fi

# ---- Done ----
echo ""
echo "====== Setup complete! ======"
echo ""
echo "To start training:"
echo "  bash train_g1_pick_apple.sh [exp_name]"
echo ""
echo "To use a custom dataset root:"
echo "  DATASET_ROOT=/path/to/datasets bash train_g1_pick_apple.sh"
