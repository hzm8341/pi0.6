#!/usr/bin/env bash
# Train an openpi JAX config until checkpoint STOP_STEP exists, then stop it (same LR schedule as a full run).
# Usage: train_until.sh <config> <exp> <stop_step> [extra train.py args]
# RESUME=1 continues an existing experiment (--resume) instead of starting over (--overwrite).
set -uo pipefail
cd "$(dirname "$0")/../.."
CFG=$1; EXP=$2; STOP=$3; shift 3
C=checkpoints/$CFG/$EXP
MODE=--overwrite; [[ ${RESUME:-0} == 1 ]] && MODE=--resume
LOG=logs/train_${CFG}_${EXP}.log
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py $CFG --exp-name $EXP $MODE --no-wandb-enabled "$@" >> $LOG 2>&1 &
TRAIN=$!
until [[ -d $C/$STOP/params ]] && ! ls -d $C/*tmp* >/dev/null 2>&1; do
    kill -0 $TRAIN 2>/dev/null || { echo "training exited early, see $LOG"; exit 1; }
    sleep 30
done
sleep 60; kill $TRAIN; wait $TRAIN 2>/dev/null; sleep 10
echo "$C/$STOP ready"
