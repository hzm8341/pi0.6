# LIBERO-10 Task 0: π0.5 SFT vs RECAP runbook

Phase 4 of `docs/RECAP_PLAN.md`. Single RTX 5090 (32GB), all training uses LoRA. The task is "put both the alphabet soup and the tomato sauce in the basket" (LIBERO-10 task 0, `task_index=5` in `physical-intelligence/libero`, 33 demos).

## Components

| Component | Purpose |
|---|---|
| `pi05_libero_lora_fewshot` | SFT baseline (K demos) |
| `pi05_recap_libero_value_lora` | value head only (`value_mode="value_only"`), LoRA backbone |
| `pi05_recap_libero_lora` | RECAP policy (advantage-conditioned), retrained from `pi05_base` |
| `AssetsConfig(assets_dir="./assets/libero_task0", asset_id="norm_stats")` | one set of norm stats for all three |
| `scripts/build_libero_recap_dataset.py` | demos (+ rollouts) → local LeRobot dataset + `recap_episodes.json` in the same frame order |
| `examples/libero/main.py` | `--task-ids`, `--episode-start`, `--record-dir`, `--summary-path`, `--no-save-videos` |
| `scripts/serve_policy.py --cfg-beta` | CFG scale for RECAP policies |
| `scripts/compute_norm_stats.py --repo-id --output-dir` | compute stats for a given dataset into a shared directory |

All three configs: 10k steps, batch 32, cosine LR (warmup 500, peak 5e-5, end 5e-6), checkpoint every 2.5k steps (all kept).

## 0. Inputs

- Task-0 demos: `/media/hzm/data24t/datasets/libero_src` (meta + 33 parquet files, `task0_episodes.json`).
- `pi05_base`: `~/.cache/openpi/openpi-assets/checkpoints/pi05_base`.
- LIBERO client image: `docker build --network host -t libero -f examples/libero/Dockerfile .` (pass the proxy as `--build-arg http_proxy=... https_proxy=...` if needed).

The client runs in Docker, the policy server uses this repo's `.venv`:

```bash
# server (host)
.venv/bin/python scripts/serve_policy.py policy:checkpoint \
    --policy.config <config> --policy.dir checkpoints/<config>/<exp>/<step> [--cfg-beta 1.5]

# client (Docker); paths are relative to the repo, mounted at /app
docker run --rm --gpus all --network host -v $PWD:/app \
    -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e MUJOCO_EGL_DEVICE_ID=0 -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e CLIENT_ARGS="--args.task-suite-name libero_10 --args.task-ids 0 ..." libero
```

Files written by the container belong to root; `sudo chown -R $USER data/libero` afterwards.

## A. Datasets and norm stats

```bash
# all 33 demos -> shared norm stats
.venv/bin/python scripts/build_libero_recap_dataset.py --repo-id local/libero_task0_demos33
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_libero_lora_fewshot \
    --repo-id local/libero_task0_demos33 --output-dir assets/libero_task0/norm_stats

# K-demo SFT dataset (start with K=10; seeded random subset)
.venv/bin/python scripts/build_libero_recap_dataset.py --repo-id local/libero_task0_demos10 --num-demos 10
```

## B. SFT

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_libero_lora_fewshot \
    --data.repo-id local/libero_task0_demos10 --exp-name k10 --overwrite
```

## C. Evaluate SFT and collect rollouts

Evaluation (50 fixed initial states):

```bash
CLIENT_ARGS="--args.task-suite-name libero_10 --args.task-ids 0 --args.num-trials-per-task 50 \
  --args.summary-path data/libero/eval/sft_k10.json --args.video-out-path data/libero/videos/sft_k10"
```

If the success rate is far from ~50%, change K in step A and retrain. Then collect 500 rollouts, cycling through the 50 initial states. Start at episode 50 so these episodes are distinct from the evaluation episodes:

```bash
CLIENT_ARGS="--args.task-suite-name libero_10 --args.task-ids 0 --args.num-trials-per-task 500 \
  --args.episode-start 50 --args.record-dir data/libero/rollouts_sft_k10 \
  --args.summary-path data/libero/eval/rollouts_sft_k10.json --args.no-save-videos"
```

## D. RECAP dataset and proxy labels

```bash
.venv/bin/python scripts/build_libero_recap_dataset.py --repo-id local/libero_task0_recap \
    --num-demos 10 --rollout-dir data/libero/rollouts_sft_k10
.venv/bin/python scripts/label_recap_advantage.py \
    --input-episodes ~/.cache/huggingface/lerobot/local/libero_task0_recap/recap_episodes.json \
    --output-dir outputs/libero_task0/labels_proxy
```

Use the same `--num-demos`/`--demo-seed` as the SFT dataset so RECAP sees the same demos. Rewards are normalized by `max_episode_length=520`, the libero_10 step limit in `main.py`.

## E. Value model and learned labels

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_recap_libero_value_lora \
    --data.recap-fields-path outputs/libero_task0/labels_proxy/lerobot_fields.npz --exp-name v1 --overwrite
.venv/bin/python scripts/label_recap_with_value.py --config-name pi05_recap_libero_value_lora \
    --checkpoint-dir checkpoints/pi05_recap_libero_value_lora/v1/9999 \
    --labels outputs/libero_task0/labels_proxy/recap_labels.jsonl \
    --recap-fields-path outputs/libero_task0/labels_proxy/lerobot_fields.npz \
    --output-dir outputs/libero_task0/labels_value --positive-fraction 0.3
```

Check `value_label_summary.json` (`value_return_spearman`, `positive_fraction_actual`).

## F. RECAP policy

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 .venv/bin/python scripts/train.py pi05_recap_libero_lora \
    --data.recap-fields-path outputs/libero_task0/labels_value/lerobot_fields.npz --exp-name r1 --overwrite
```

## G. Evaluation and ablations

- RECAP: serve with `--cfg-beta 1.0`, `1.5`, `2.0`; 50 episodes each (`--args.summary-path data/libero/eval/recap_b<β>.json`).
- Filtered BC: build `--repo-id local/libero_task0_filtered` from the demos plus only the successful rollouts (copy them into a separate directory), then train `pi05_libero_lora_fewshot` on it.
- Report: success rate, mean steps of successful episodes, label distribution.
