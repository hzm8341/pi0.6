"""Print the positive-label share by data source (demo / successful rollout / failed rollout).

Usage: python scripts/libero/label_sources.py <recap_episodes.json> <lerobot_fields.npz>
"""

import json
import pathlib
import sys

import numpy as np

episodes = json.loads(pathlib.Path(sys.argv[1]).read_text())
ind = np.load(sys.argv[2])["advantage_indicator"].astype(bool)
cat = np.concatenate(
    [
        np.full(len(e["frames"]), "demo" if e["metadata"]["source"] == "demo" else ("succ" if e["success"] else "fail"))
        for e in episodes
    ]
)
for c in ("demo", "succ", "fail"):
    m = cat == c
    print(f"{c}: frames={m.sum()} positive={ind[m].mean():.1%} share_of_positive={ind[m].sum() / ind.sum():.1%}")
