#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME="${CONFIG_NAME:-pi05_recap}"
EPISODES_DIR="${EPISODES_DIR:-}"
LABEL_DIR="${LABEL_DIR:-outputs/recap_labels}"
TRIMMED_DIR="${TRIMMED_DIR:-outputs/recap_trimmed_episodes}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${LABEL_DIR}/split_manifest.json}"
OFFLINE_SUMMARY="${OFFLINE_SUMMARY:-${LABEL_DIR}/offline_eval_summary.json}"
RECAP_FIELDS_PATH="${RECAP_FIELDS_PATH:-${LABEL_DIR}/lerobot_fields.npz}"
POSITIVE_FRACTION="${POSITIVE_FRACTION:-0.4}"
N_STEP_LOOKAHEAD="${N_STEP_LOOKAHEAD:-50}"
EVAL_FRACTION="${EVAL_FRACTION:-0.15}"
SPLIT_SEED="${SPLIT_SEED:-0}"
ACTION_NORM_THRESHOLD="${ACTION_NORM_THRESHOLD:-0.05}"
STATIC_ACTION_THRESHOLD="${STATIC_ACTION_THRESHOLD:-0.05}"
MIN_FRAMES="${MIN_FRAMES:-10}"
SMOKE_STEPS="${SMOKE_STEPS:-10}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"
EXP_NAME="${EXP_NAME:-test_tube_recap_v1}"
RUN_FULL_TRAIN="${RUN_FULL_TRAIN:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-False}"

if [[ -z "${EPISODES_DIR}" ]]; then
  echo "ERROR: EPISODES_DIR must point to a RECAP episode JSON file or directory." >&2
  exit 2
fi

mkdir -p "${LABEL_DIR}" "${TRIMMED_DIR}"

echo "==> Creating train/eval split manifest"
PYTHONPATH=src uv run python scripts/create_recap_split_manifest.py \
  --input-episodes "${EPISODES_DIR}" \
  --output "${SPLIT_MANIFEST}" \
  --eval-fraction "${EVAL_FRACTION}" \
  --seed "${SPLIT_SEED}"

echo "==> Trimming static tail frames"
PYTHONPATH=src uv run python scripts/trim_recap_episodes.py \
  --input-episodes "${EPISODES_DIR}" \
  --output-episodes "${TRIMMED_DIR}" \
  --action-norm-threshold "${ACTION_NORM_THRESHOLD}" \
  --min-frames "${MIN_FRAMES}"

echo "==> Writing offline episode summary"
PYTHONPATH=src uv run python scripts/eval_recap_episodes.py \
  --input-episodes "${TRIMMED_DIR}" \
  --output "${OFFLINE_SUMMARY}" \
  --static-action-threshold "${STATIC_ACTION_THRESHOLD}"

echo "==> Generating RECAP advantage labels"
PYTHONPATH=src uv run python scripts/label_recap_advantage.py \
  --input-episodes "${TRIMMED_DIR}" \
  --output-dir "${LABEL_DIR}" \
  --positive-fraction "${POSITIVE_FRACTION}" \
  --n-step-lookahead "${N_STEP_LOOKAHEAD}"

echo "==> Verifying RECAP sidecar arrays"
PYTHONPATH=src uv run python - "${RECAP_FIELDS_PATH}" <<'PY'
import sys
import numpy as np

path = sys.argv[1]
fields = np.load(path)
required = ("advantage_indicator", "use_advantage", "is_human_intervention")
lengths = {}
for key in required:
    if key not in fields.files:
        raise SystemExit(f"missing required field: {key}")
    values = fields[key]
    if values.dtype != np.bool_:
        raise SystemExit(f"{key} must be bool, got {values.dtype}")
    lengths[key] = len(values)
if len(set(lengths.values())) != 1:
    raise SystemExit(f"field lengths differ: {lengths}")
print(f"sidecar ok: {lengths}")
PY

echo "==> Running focused RECAP tests"
PYTHONPATH=src uv run pytest \
  src/openpi/training/test_recap_dataset_tools.py \
  src/openpi/training/test_recap_episode_io.py \
  src/openpi/training/test_recap_offline.py \
  src/openpi/training/test_recap_collector.py \
  src/openpi/training/test_recap_train_config.py \
  src/openpi/training/data_loader_test.py::test_recap_fields_dataset_merges_sidecar_fields \
  src/openpi/training/data_loader_test.py::test_recap_fields_dataset_rejects_length_mismatch \
  -q

echo "==> Computing normalization stats for ${CONFIG_NAME}"
uv run scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"

echo "==> Running smoke training"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
uv run scripts/train.py "${CONFIG_NAME}" \
  --data.recap-fields-path "${RECAP_FIELDS_PATH}" \
  --exp-name "${EXP_NAME}_smoke" \
  --num-train-steps "${SMOKE_STEPS}" \
  --batch-size "${SMOKE_BATCH_SIZE}" \
  --overwrite \
  --wandb-enabled "${WANDB_ENABLED}"

if [[ "${RUN_FULL_TRAIN}" == "1" ]]; then
  echo "==> Running full training"
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
  uv run scripts/train.py "${CONFIG_NAME}" \
    --data.recap-fields-path "${RECAP_FIELDS_PATH}" \
    --exp-name "${EXP_NAME}" \
    --overwrite \
    --wandb-enabled "${WANDB_ENABLED}"
else
  echo "==> Smoke validation complete. Set RUN_FULL_TRAIN=1 to start full training."
fi
