#!/usr/bin/env bash
# Phase 4 steps D-E: RECAP dataset (same 10 demos as SFT + rollouts) -> proxy labels -> value model.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROLLOUTS=${ROLLOUTS:-data/libero/rollouts_sft_k10}
REPO=${REPO:-local/libero_task0_recap}
OUT=${OUT:-outputs/libero_task0}
.venv/bin/python scripts/build_libero_recap_dataset.py --repo-id "$REPO" --num-demos 10 --rollout-dir "$ROLLOUTS" --overwrite
.venv/bin/python scripts/label_recap_advantage.py \
    --input-episodes "$HOME/.cache/huggingface/lerobot/$REPO/recap_episodes.json" --output-dir "$OUT/labels_proxy"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_recap_libero_value_lora \
    --data.repo-id "$REPO" --data.recap-fields-path "$OUT/labels_proxy/lerobot_fields.npz" \
    --exp-name v1 --overwrite --no-wandb-enabled
