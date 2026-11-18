#!/usr/bin/env bash
# Phase 4 steps F-G: RECAP policy (stopped at STOP_STEP like the SFT baseline, same LR schedule) -> eval at several betas.
set -uo pipefail
cd "$(dirname "$0")/../.."
LABELS=${LABELS:-outputs/libero_task0/labels_value_s2500/lerobot_fields.npz}
EXP=${EXP:-r1}; STOP_STEP=${STOP_STEP:-2500}
C=checkpoints/pi05_recap_libero_lora/$EXP
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_recap_libero_lora \
    --data.recap-fields-path "$LABELS" --exp-name "$EXP" --overwrite --no-wandb-enabled > logs/train_recap_$EXP.log 2>&1 &
TRAIN=$!
until [[ -d $C/$STOP_STEP/params ]] && ! ls -d $C/*tmp* >/dev/null 2>&1; do
    kill -0 $TRAIN 2>/dev/null || { echo "training exited early, see logs/train_recap_$EXP.log"; exit 1; }
    sleep 30
done
sleep 60; kill $TRAIN; wait $TRAIN 2>/dev/null; sleep 10
scripts/libero/eval_checkpoint.sh pi05_recap_libero_lora $C/$STOP_STEP recap_${EXP}_s${STOP_STEP}_b1.0
for B in 1.5 2.0; do
    scripts/libero/eval_checkpoint.sh pi05_recap_libero_lora $C/$STOP_STEP recap_${EXP}_s${STOP_STEP}_b$B --cfg-beta $B
done
