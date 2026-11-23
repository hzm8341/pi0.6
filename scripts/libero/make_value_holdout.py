"""Hold out a fraction of rollout episodes from value training by setting their value_target to NaN.

The value loss masks NaN targets (Pi0._value_cross_entropy), so the trained value head never fits those episodes;
`eval_value_holdout.py` then measures the value and the advantage labels on them. Demos always stay in training.
The split is stratified by success.

Usage: python scripts/libero/make_value_holdout.py <labels_dir> <out_dir> [--frac 0.2] [--seed 0]
Writes <out_dir>/lerobot_fields.npz, recap_labels.jsonl (copied) and holdout_episodes.json.
"""

import argparse
import json
import pathlib
import shutil

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("labels_dir", type=pathlib.Path)
ap.add_argument("out_dir", type=pathlib.Path)
ap.add_argument("--frac", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

records = [json.loads(line) for line in (a.labels_dir / "recap_labels.jsonl").open()]
fields = dict(np.load(a.labels_dir / "lerobot_fields.npz"))
assert len(records) == len(fields["value_target"])
ep_ids = np.asarray([r["episode_id"] for r in records])
episodes = list(dict.fromkeys(ep_ids))
success = {r["episode_id"]: r["success"] for r in records}
rng = np.random.default_rng(a.seed)
held = []
for s in (True, False):
    pool = [e for e in episodes if not e.startswith("demo") and success[e] == s]
    held += list(rng.choice(pool, size=round(a.frac * len(pool)), replace=False))
mask = np.isin(ep_ids, held)
fields["value_target"] = np.where(mask, np.nan, fields["value_target"]).astype(np.float32)
a.out_dir.mkdir(parents=True, exist_ok=True)
np.savez(a.out_dir / "lerobot_fields.npz", **fields)
shutil.copy(a.labels_dir / "recap_labels.jsonl", a.out_dir / "recap_labels.jsonl")
(a.out_dir / "holdout_episodes.json").write_text(json.dumps(sorted(map(str, held)), indent=1))
n_succ = sum(success[e] for e in held)
print(f"held out {len(held)} episodes ({n_succ} success, {len(held) - n_succ} fail), {mask.sum()} frames")
