"""Relabel RECAP advantages with a learned value head.

Pipeline (see docs/RECAP_PLAN.md, Phase 2):
  1. `scripts/label_recap_advantage.py` → recap_labels.jsonl + lerobot_fields.npz (with value_target)
  2. train a value checkpoint, e.g. `scripts/train.py pi05_recap_value --data.recap-fields-path .../lerobot_fields.npz`
  3. this script: predict V(o_t) for every frame of the training dataset, recompute N-step advantages,
     threshold, and write a new recap_labels.jsonl + lerobot_fields.npz for `pi05_recap` training.

The training dataset (built from `--config-name`) must have the same flattened frame order as the
records in `--labels`, exactly like the sidecar contract used during training.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch.utils.data
import tyro

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training.recap_offline import load_label_records, relabel_with_values, write_recap_label_outputs

logger = logging.getLogger("recap.value_label")


@dataclasses.dataclass
class Args:
    # Train config whose model has `ReCAPConfig(value_head=True)`, e.g. "pi05_recap_value".
    config_name: str
    # Checkpoint step directory, e.g. checkpoints/pi05_recap_value/exp/29999 (must contain `params/`).
    checkpoint_dir: Path
    # recap_labels.jsonl from the proxy labeling step (rewards, episode ids, intervention flags).
    labels: Path
    # Sidecar used to build the dataset order (normally the same lerobot_fields.npz used for value training).
    recap_fields_path: Path
    output_dir: Path
    # LeRobot repo id; defaults to the config's.
    repo_id: str | None = None
    positive_fraction: float = 0.3
    n_step_lookahead: int = 50
    batch_size: int = 32
    num_workers: int = 4
    # Only process the first N frames (smoke tests). Must be None for real labeling.
    max_frames: int | None = None


def _collate(items):
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *items)


def predict_values(args: Args) -> np.ndarray:
    config = _config.get_config(args.config_name)
    recap = getattr(config.model, "recap", None)
    if recap is None or not (recap.enabled and recap.value_head):
        raise ValueError(f"Config {args.config_name!r} has no RECAP value head.")

    data_factory = config.data
    overrides = {"recap_fields_path": str(args.recap_fields_path)}
    if args.repo_id is not None:
        overrides["repo_id"] = args.repo_id
    data_factory = dataclasses.replace(data_factory, **overrides)
    data_config = data_factory.create(config.assets_dirs, config.model)

    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.transform_dataset(dataset, data_config)
    if args.max_frames is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_frames, len(dataset))))

    params = _model.restore_params(args.checkpoint_dir / "params", dtype=jnp.bfloat16)
    model = config.model.load(params)

    @jax.jit
    def value_fn(batch):
        observation = _model.Observation.from_dict(batch)
        values, _ = model.predict_value(observation)
        return values

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=_collate
    )
    values = []
    for step, batch in enumerate(loader):
        batch = {k: v for k, v in batch.items() if k != "actions"}
        values.append(np.asarray(value_fn(batch)))
        if step % 50 == 0:
            logger.info("predicted %d / %d frames", sum(len(v) for v in values), len(dataset))
    return np.concatenate(values)


def main(args: Args) -> None:
    records = load_label_records(args.labels)
    values = predict_values(args)
    if args.max_frames is not None:
        records = records[: len(values)]
    result = relabel_with_values(
        records, values, positive_fraction=args.positive_fraction, n_step_lookahead=args.n_step_lookahead
    )
    write_recap_label_outputs(result, args.output_dir)

    returns = np.asarray([r["return"] for r in records], dtype=np.float32)
    summary = {
        "frames": len(records),
        "positive_fraction_actual": float(result.fields["advantage_indicator"].mean()),
        "value_return_spearman": _spearman(values, returns),
        "value_mae": float(np.mean(np.abs(values - returns))),
    }
    (args.output_dir / "value_label_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("summary: %s", summary)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    main(tyro.cli(Args))
