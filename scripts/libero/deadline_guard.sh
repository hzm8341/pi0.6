#!/usr/bin/env bash
# Stop this project's GPU jobs at DEADLINE (default 2026-11-21 00:45) so the GPU is free by 01:00.
DEADLINE=${DEADLINE:-2026-11-21 00:45}
sleep $(( $(date -d "$DEADLINE" +%s) - $(date +%s) ))
date '+deadline guard fired at %F %H:%M'
for pat in "scripts/train.py pi05_" "scripts/train_pytorch.py" "scripts/serve_policy.py" "scripts/libero/run_" "scripts/libero/full_ft_trial" "scripts/libero/eval_checkpoint"; do
    pgrep -f "$pat" | xargs -r kill
done
docker ps -q --filter ancestor=libero | xargs -r docker kill
sleep 20; nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
