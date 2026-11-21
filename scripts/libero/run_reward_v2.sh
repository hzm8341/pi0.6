#!/usr/bin/env bash
# Reward redesign round (docs/DECISIONS.md 2026-11-21): separated rewards (failure = 520 extra steps, not clipped)
# -> value v2 @2.5k -> learned labels -> RECAP r3 @2.5k -> 200-trial evals (4 x 50 initial states).
# Baselines get episodes 50-199 added to their existing 0-49 results.
set -uo pipefail
cd "$(dirname "$0")/../.."
L=scripts/libero; O=outputs/libero_task0; E=data/libero/eval
EPS=$HOME/.cache/huggingface/lerobot/local/libero_task0_recap/recap_episodes.json
date '+value v2 start %F %H:%M'
$L/train_until.sh pi05_recap_libero_value_lora v2 2500 --data.recap-fields-path $O/labels_proxy_sep/lerobot_fields.npz || exit 1
.venv/bin/python scripts/label_recap_with_value.py --config-name pi05_recap_libero_value_lora \
    --checkpoint-dir checkpoints/pi05_recap_libero_value_lora/v2/2500 --labels $O/labels_proxy_sep/recap_labels.jsonl \
    --recap-fields-path $O/labels_proxy_sep/lerobot_fields.npz --output-dir $O/labels_value_sep_s2500 \
    --positive-fraction 0.3 > logs/label_value_sep.log 2>&1 || { echo "labeling failed"; exit 1; }
cat $O/labels_value_sep_s2500/value_label_summary.json
.venv/bin/python $L/label_sources.py $EPS $O/labels_value_sep_s2500/lerobot_fields.npz
date '+RECAP r3 start %F %H:%M'
$L/train_until.sh pi05_recap_libero_lora r3 2500 --data.recap-fields-path $O/labels_value_sep_s2500/lerobot_fields.npz || exit 1
C=checkpoints/pi05_recap_libero_lora/r3/2500
NUM_TRIALS=200 $L/eval_checkpoint.sh pi05_recap_libero_lora $C recap_r3_s2500_b1.0_n200 | tail -1
NUM_TRIALS=200 $L/eval_checkpoint.sh pi05_recap_libero_lora $C recap_r3_s2500_b1.5_n200 --cfg-beta 1.5 | tail -1
# Baselines: add episodes 50-199.
ext() {  # tag config ckpt [serve args]
    local tag=$1 cfg=$2 ckpt=$3; shift 3
    EPISODE_START=50 NUM_TRIALS=150 $L/eval_checkpoint.sh $cfg $ckpt ${tag}_ep50_199 "$@" | tail -1
    .venv/bin/python $L/merge_evals.py $E/${tag}_n200.json $E/$tag.json $E/${tag}_ep50_199.json
}
ext filtbc_s2500 pi05_libero_lora_fewshot checkpoints/pi05_libero_lora_fewshot/filt/2500
ext recap_r2_s2500_b1.5 pi05_recap_libero_lora checkpoints/pi05_recap_libero_lora/r2/2500 --cfg-beta 1.5
ext sft_k10_s2500 pi05_libero_lora_fewshot checkpoints/pi05_libero_lora_fewshot/k10/2500
for t in recap_r3_s2500_b1.0_n200 recap_r3_s2500_b1.5_n200; do .venv/bin/python $L/merge_evals.py $E/${t}_m.json $E/$t.json; done
date '+reward v2 round done %F %H:%M'
