# RECAP GPU Training Runbook

This guide describes how to validate and start `pi05_recap` training on a GPU server for the test-tube/rack task.

## 1. Upload Required Inputs

Upload or make available on the GPU server:

- The repository checkout containing this branch.
- The cleaned LeRobot dataset.
- RECAP episode JSON files with `success`, `frames`, `observation`, `action`, and optional `is_human_intervention`, or a LeRobot dataset that can be converted first.
- Enough disk space for `outputs/`, `assets/`, and checkpoints.

Do not start long training before the smoke run passes.

## 2. Prepare Environment

```bash
cd /path/to/pi0.6
git status -sb
uv sync
```

If you use Hugging Face-hosted datasets, log in before training:

```bash
hf auth login
```

If you use GCS checkpoints and need authentication, configure that before training as well.

## 3. Convert LeRobot Dataset If Needed

If you only have a local LeRobot dataset, convert it to RECAP episode JSON first:

```bash
PYTHONPATH=src python scripts/convert_lerobot_to_recap_episodes.py \
  --lerobot-root /path/to/lerobot_dataset \
  --output-episodes outputs/recap_episodes \
  --default-success unknown
```

Use `--max-episodes 2` for a quick smoke conversion. With `--default-success unknown`, generated episodes are marked with `metadata.success_needs_review=true`; review success/failure labels before serious training. You can also pass `--success-labels /path/to/success_labels.jsonl`, where each row contains `episode_id` or `episode_index` plus `success`.

## 4. Run One-Command Validation

Set `EPISODES_DIR` to a RECAP episode JSON file or directory.

```bash
EPISODES_DIR=outputs/recap_episodes \
CONFIG_NAME=pi05_recap \
EXP_NAME=test_tube_recap_v1 \
WANDB_ENABLED=False \
bash scripts/run_recap_gpu_validation.sh
```

The script will:

1. Create `outputs/recap_labels/split_manifest.json`.
2. Trim static tail frames into `outputs/recap_trimmed_episodes`.
3. Write `outputs/recap_labels/offline_eval_summary.json`.
4. Generate `outputs/recap_labels/recap_labels.jsonl`.
5. Generate `outputs/recap_labels/lerobot_fields.npz`.
6. Check the RECAP sidecar field names, dtypes, and internal lengths.
7. Run focused RECAP tests.
8. Compute normalization statistics.
9. Run a 10-step `pi05_recap` smoke training job with the sidecar fields.

If this fails, fix that error before running a long training job.

## 5. Start Full Training

After the smoke run passes:

```bash
EPISODES_DIR=outputs/recap_episodes \
CONFIG_NAME=pi05_recap \
EXP_NAME=test_tube_recap_v1 \
WANDB_ENABLED=True \
RUN_FULL_TRAIN=1 \
bash scripts/run_recap_gpu_validation.sh
```

If you already generated labels and only want to train:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_recap \
  --data.recap-fields-path outputs/recap_labels/lerobot_fields.npz \
  --exp-name test_tube_recap_v1 \
  --overwrite
```

The `lerobot_fields.npz` arrays must match the flattened frame order and length of the LeRobot dataset used by the dataloader.

## 6. Common Failures

- `RECAP field ... must have the same length as the dataset`: the sidecar arrays do not match the LeRobot flattened frame dataset. Regenerate labels from the same trimmed dataset that training reads, or implement permanent LeRobot write-back for that dataset.
- `Prompt is required`: the dataset transform did not provide a prompt. Set the right data config or provide prompt fields.
- CUDA/JAX OOM: lower batch size first, then adjust `XLA_PYTHON_CLIENT_MEM_FRACTION`, then consider FSDP settings.
- Video decode errors: verify the LeRobot videos, timestamps, and `lerobot_video_tolerance_s`.
- Smoke loss is NaN: inspect action normalization stats, action range, and gripper channel values before continuing.

## 7. Before Robot Testing

Do not run the full insertion task immediately after training. First inspect:

- Offline action range and continuity.
- Gripper open/close timing.
- Static action ratio.
- Failure examples.
- Short, low-speed robot stages: approach only, approach + pose, grasp + lift, then move near the device.

The final insertion stage should keep visual servoing and force/impedance protection.
