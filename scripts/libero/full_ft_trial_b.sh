#!/usr/bin/env bash
# Single-5090 full fine-tuning trial, plan B: bf16 weights/grads on the GPU, fp32 master + AdamW moments in CPU RAM
# (CPUOffloadAdamW). Uses the real config (batch 32 = 8 x 4) for STEPS optimizer steps.
# Rules (docs/DECISIONS.md 2026-11-20): wait while Abaqus runs; needs >= 70GB available RAM; GPU must be idle.
# Logs: logs/full_ft_trial_planb.log, GPU samples logs/full_ft_trial_planb_nvsmi.csv, RAM samples ..._ram.csv.
set -uo pipefail
cd "$(dirname "$0")/../.."
CFG=pi05_libero_task0_full_pytorch
STEPS=${STEPS:-12}
TORCH_CU128=${TORCH_CU128:-/media/hzm/data24t/envs/torch_cu128}
[[ -d $TORCH_CU128 ]] && export PYTHONPATH=$TORCH_CU128${PYTHONPATH:+:$PYTHONPATH}
abaqus_running() { pgrep -x 'standard|explicit|pre|package|abaqus|ABQcaeK|ABQcaeG|ABQSMAStdCore|ABQSMAExpCore' >/dev/null; }
while abaqus_running; do date '+Abaqus running, waiting (%F %H:%M)'; sleep 300; done
avail=$(free -g | awk '/^Mem:/{print $7}')
if (( avail < 70 )); then echo "only ${avail}GB RAM available (< 70), abort"; exit 2; fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then echo "GPU busy, abort"; exit 1; fi
free -g
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 2 > logs/full_ft_trial_planb_nvsmi.csv &
SMI=$!
( while true; do echo "$(date +%T),$(free -m | awk '/^Mem:/{print $3","$7}')"; sleep 2; done ) > logs/full_ft_trial_planb_ram.csv &
RAM=$!
trap 'kill $SMI $RAM 2>/dev/null' EXIT
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/train_pytorch.py $CFG \
    --pytorch-optimizer-offload --num-train-steps $STEPS --log-interval 1 --save-interval 1000000 \
    --exp-name trial_planb --overwrite --no-wandb-enabled --checkpoint-base-dir /tmp/full_ft_trial_ckpt \
    > logs/full_ft_trial_planb.log 2>&1
echo "planb exit=$?"
rm -rf /tmp/full_ft_trial_ckpt
grep -E "peak_allocated|step=|optimizer step took|Error|OutOfMemory" logs/full_ft_trial_planb.log | tail -25
echo "peak GPU MiB: $(awk -F', ' 'NR>1{gsub(/ MiB/,"",$2); if($2>m)m=$2} END{print m}' logs/full_ft_trial_planb_nvsmi.csv)"
echo "peak RAM used MiB / min available MiB: $(awk -F, '{if($2>u)u=$2; if(m==""||$3<m)m=$3} END{print u" / "m}' logs/full_ft_trial_planb_ram.csv)"
