"""Diagnostic relabeling: keep value-based advantage labels only inside successful episodes.

Frames of failed episodes become "Advantage: negative". Tests whether RECAP r1 lost to filtered BC because 41% of its
positive frames came from failed rollouts. Usage:
  python scripts/libero/gate_labels_by_success.py <recap_episodes.json> <in lerobot_fields.npz> <out dir>
"""

import json
import pathlib
import sys

import numpy as np

episodes_path, in_path, out_dir = sys.argv[1:4]
episodes = json.loads(pathlib.Path(episodes_path).read_text())
fields = dict(np.load(in_path))
success = np.concatenate([np.full(len(e["frames"]), bool(e["success"])) for e in episodes])
if len(success) != len(fields["advantage_indicator"]):
    raise ValueError("episode file and sidecar have different frame counts")
before = fields["advantage_indicator"].astype(bool)
fields["advantage_indicator"] = (before & success).astype(fields["advantage_indicator"].dtype)
out = pathlib.Path(out_dir)
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "lerobot_fields.npz", **fields)
summary = {
    "source": in_path,
    "positive_fraction_before": float(before.mean()),
    "positive_fraction_after": float(fields["advantage_indicator"].astype(bool).mean()),
}
(out / "gate_summary.json").write_text(json.dumps(summary, indent=2))
print(summary)
