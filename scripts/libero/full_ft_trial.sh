#!/usr/bin/env bash
# Single-5090 full fine-tuning trial (docs/pi05全量训练_单卡5090方案.md section 3, plan A: pure bf16).
# 1) convert pi05_base to PyTorch, 2) 60 steps at micro-batch 4 (no accumulation), 3) 10 steps at 8 = 2 x 4 to
# check gradient accumulation. Logs: logs/full_ft_trial_*.log, GPU memory samples: logs/full_ft_trial_nvsmi.csv.
set -uo pipefail
cd "$(dirname "$0")/../.."
CFG=pi05_libero_task0_full_pytorch
# .venv torch (cu126) has no sm_120 kernels; prefer the cu128 build installed separately (docs/DECISIONS.md 2026-11-20).
TORCH_CU128=${TORCH_CU128:-/media/hzm/data24t/envs/torch_cu128}
[[ -d $TORCH_CU128 ]] && export PYTHONPATH=$TORCH_CU128${PYTHONPATH:+:$PYTHONPATH}
W=$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then echo "GPU busy, abort"; nvidia-smi; exit 1; fi
free -g
if [[ ! -f $W/model.safetensors ]]; then
    .venv/bin/python examples/convert_jax_model_to_pytorch.py --config-name $CFG \
        --checkpoint-dir $HOME/.cache/openpi/openpi-assets/checkpoints/pi05_base --output-path $W --precision float32 \
        > logs/full_ft_trial_convert.log 2>&1 || { echo "conversion failed"; tail -20 logs/full_ft_trial_convert.log; exit 1; }
fi
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2 > logs/full_ft_trial_nvsmi.csv &
SMI=$!
trap 'kill $SMI 2>/dev/null' EXIT
run() {  # tag batch accum steps
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/train_pytorch.py $CFG \
        --batch-size $2 --gradient-accumulation-steps $3 --num-train-steps $4 --log-interval 5 --save-interval 1000000 \
        --exp-name trial_$1 --overwrite --no-wandb-enabled --checkpoint-base-dir /tmp/full_ft_trial_ckpt > logs/full_ft_trial_$1.log 2>&1
    echo "trial_$1 exit=$?"; grep -E "peak_allocated|step=|Error|OutOfMemory" logs/full_ft_trial_$1.log | tail -15
    rm -rf /tmp/full_ft_trial_ckpt
}
run b4 4 1 60
run b8a2 8 2 10
echo "peak GPU memory used (nvidia-smi, MiB):"; awk -F', ' 'NR>1{gsub(/ MiB/,"",$2); if($2>m)m=$2} END{print m}' logs/full_ft_trial_nvsmi.csv
