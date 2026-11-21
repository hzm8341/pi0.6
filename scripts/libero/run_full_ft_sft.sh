#!/usr/bin/env bash
# Full fine-tuning SFT on LIBERO Task 0 (10 demos, same data/LR as the LoRA SFT) with plan B (CPUOffloadAdamW),
# trained until checkpoint STOP_STEP, then evaluated 200 times. RESUME=1 continues the experiment.
# Rules (docs/DECISIONS.md 2026-11-20): wait while Abaqus runs; >= 70GB available RAM; GPU idle.
set -uo pipefail
cd "$(dirname "$0")/../.."
CFG=pi05_libero_task0_full_pytorch; EXP=${EXP:-full_k10}; STOP=${STOP_STEP:-2500}
C=checkpoints/$CFG/$EXP
export PYTHONPATH=/media/hzm/data24t/envs/torch_cu128${PYTHONPATH:+:$PYTHONPATH}  # sm_120 kernels
abaqus_running() { pgrep -x 'standard|explicit|pre|package|abaqus|ABQcaeK|ABQcaeG|ABQSMAStdCore|ABQSMAExpCore' >/dev/null; }
while abaqus_running; do date '+Abaqus running, waiting (%F %H:%M)'; sleep 300; done
until (( $(free -g | awk '/^Mem:/{print $7}') >= 70 )) && [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
    date '+waiting for RAM >= 70GB and an idle GPU (%F %H:%M)'; sleep 300
done
date '+full FT start %F %H:%M'; free -g | sed -n 2p
MODE=--overwrite; [[ ${RESUME:-0} == 1 ]] && MODE=--resume
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/train_pytorch.py $CFG --exp-name $EXP $MODE \
    --no-wandb-enabled >> logs/train_full_ft_$EXP.log 2>&1 &
TRAIN=$!
until [[ -f $C/$STOP/model.safetensors && -f $C/$STOP/metadata.pt ]]; do
    kill -0 $TRAIN 2>/dev/null || { echo "training exited early, see logs/train_full_ft_$EXP.log"; exit 1; }
    sleep 60
done
sleep 30; kill $TRAIN; wait $TRAIN 2>/dev/null; sleep 10
date '+full FT stopped at %F %H:%M'
L=scripts/libero
NUM_TRIALS=200 $L/eval_checkpoint.sh $CFG $C/$STOP full_k10_s${STOP}_n200 | tail -1
.venv/bin/python $L/merge_evals.py data/libero/eval/full_k10_s${STOP}_n200_m.json data/libero/eval/full_k10_s${STOP}_n200.json
