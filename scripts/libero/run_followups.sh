#!/usr/bin/env bash
# Follow-up experiments (docs/DECISIONS.md 2026-11-20): filtered BC @2.5k, then RECAP r1 continued to 5k.
set -uo pipefail
cd "$(dirname "$0")/../.."
L=scripts/libero
date '+filtered BC start %F %H:%M'
$L/train_until.sh pi05_libero_lora_fewshot filt 2500 --data.repo-id local/libero_task0_filtered \
    && $L/eval_checkpoint.sh pi05_libero_lora_fewshot checkpoints/pi05_libero_lora_fewshot/filt/2500 filtbc_s2500
date '+RECAP 5k start %F %H:%M'
RESUME=1 $L/train_until.sh pi05_recap_libero_lora r1 5000 \
    --data.recap-fields-path outputs/libero_task0/labels_value_s2500/lerobot_fields.npz || exit 1
C=checkpoints/pi05_recap_libero_lora/r1/5000
$L/eval_checkpoint.sh pi05_recap_libero_lora $C recap_r1_s5000_b1.0
for B in ${BETAS:-1.5 2.0}; do $L/eval_checkpoint.sh pi05_recap_libero_lora $C recap_r1_s5000_b$B --cfg-beta $B; done
date '+followups done %F %H:%M'
