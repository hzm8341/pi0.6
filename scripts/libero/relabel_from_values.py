"""Recompute RECAP advantage labels from the per-frame values already stored in a recap_labels.jsonl.

No model inference: reuses `value` from the output of label_recap_with_value.py and only changes the advantage
estimator. --n-step 0 means Monte Carlo (A = G_t - V(o_t)).

Usage: python scripts/libero/relabel_from_values.py <labels_dir_with_values> <out_dir> --n-step 200
"""

import argparse
import pathlib

import numpy as np

from openpi.training.recap_offline import load_label_records
from openpi.training.recap_offline import relabel_with_values
from openpi.training.recap_offline import write_recap_label_outputs

ap = argparse.ArgumentParser()
ap.add_argument("labels_dir", type=pathlib.Path)
ap.add_argument("out_dir", type=pathlib.Path)
ap.add_argument("--n-step", type=int, required=True)
ap.add_argument("--positive-fraction", type=float, default=0.3)
a = ap.parse_args()

records = load_label_records(a.labels_dir / "recap_labels.jsonl")
values = np.asarray([r["value"] for r in records], dtype=np.float32)
result = relabel_with_values(
    records, values, positive_fraction=a.positive_fraction, n_step_lookahead=a.n_step or None
)
write_recap_label_outputs(result, a.out_dir)
ind = result.fields["advantage_indicator"]
print(f"wrote {len(records)} frames to {a.out_dir}, positive {ind.mean():.1%}")
