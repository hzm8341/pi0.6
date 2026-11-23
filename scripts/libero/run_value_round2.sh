#!/usr/bin/env bash
# Value diagnosis round (docs/DECISIONS.md 2026-11-23):
#  1. held-out value: train value on labels_proxy_sep with 100 rollouts (20%) masked (holdout_sep), predict all
#     frames, report value + advantage-estimator metrics on the held-out episodes;
#  2. RECAP with longer-horizon advantages from the full value v2: r4 (N=200), r5 (Monte Carlo), 2.5k steps each,
#     200-trial evals at beta 1.0 / 1.5.
# Waits for the GPU to be free first. Rerunnable: finished checkpoints are not retrained. Results: logs/run_value_round2.log.
set -uo pipefail
cd "$(dirname "$0")/../.."
L=scripts/libero; O=outputs/libero_task0; E=data/libero/eval
EPS=$HOME/.cache/huggingface/lerobot/local/libero_task0_recap/recap_episodes.json
for t in n200 mc; do echo "labels $t:"; .venv/bin/python $L/label_sources.py $EPS $O/labels_value_sep_s2500_$t/lerobot_fields.npz; done
$L/wait_gpu_free.sh
date '+value holdout start %F %H:%M'
[[ -d checkpoints/pi05_recap_libero_value_lora/v2h/2500/params ]] || $L/train_until.sh pi05_recap_libero_value_lora v2h 2500 --data.recap-fields-path $O/holdout_sep/lerobot_fields.npz || exit 1
.venv/bin/python scripts/label_recap_with_value.py --config-name pi05_recap_libero_value_lora \
    --checkpoint-dir checkpoints/pi05_recap_libero_value_lora/v2h/2500 --labels $O/holdout_sep/recap_labels.jsonl \
    --recap-fields-path $O/holdout_sep/lerobot_fields.npz --output-dir $O/labels_value_sep_h2500 \
    --positive-fraction 0.3 > logs/label_value_sep_h2500.log 2>&1 || { echo "labeling failed"; exit 1; }
.venv/bin/python $L/eval_value_holdout.py $O/labels_value_sep_h2500 $O/holdout_sep/holdout_episodes.json \
    --out $O/labels_value_sep_h2500/holdout_eval.json
for r in "r4 n200" "r5 mc"; do
    set -- $r
    $L/wait_gpu_free.sh
    date "+RECAP $1 ($2) start %F %H:%M"
    [[ -d checkpoints/pi05_recap_libero_lora/$1/2500/params ]] || $L/train_until.sh pi05_recap_libero_lora $1 2500 --data.recap-fields-path $O/labels_value_sep_s2500_$2/lerobot_fields.npz || exit 1
    C=checkpoints/pi05_recap_libero_lora/$1/2500
    for b in 1.0 1.5; do
        t=recap_$1_s2500_b${b}_n200
        NUM_TRIALS=200 $L/eval_checkpoint.sh pi05_recap_libero_lora $C $t --cfg-beta $b | tail -1
        .venv/bin/python $L/merge_evals.py $E/${t}_m.json $E/$t.json
    done
done
date '+value round 2 done %F %H:%M'
