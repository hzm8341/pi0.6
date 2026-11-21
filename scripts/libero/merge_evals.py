"""Merge eval summaries (e.g. episodes 0-49 and 50-199) and print success rate with standard error.

Usage: python scripts/libero/merge_evals.py <out.json> <in1.json> [<in2.json> ...]
"""

import json
import math
import pathlib
import sys

episodes = {}
for path in sys.argv[2:]:
    for e in json.loads(pathlib.Path(path).read_text())["episodes"]:
        episodes[e["episode_idx"]] = e
eps = [episodes[k] for k in sorted(episodes)]
n, s = len(eps), sum(e["success"] for e in eps)
p = s / n
steps = [e["num_steps"] for e in eps if e["success"]]
out = {
    "sources": sys.argv[2:],
    "total_episodes": n,
    "total_successes": s,
    "success_rate": p,
    "stderr": math.sqrt(p * (1 - p) / n),
    "mean_success_steps": sum(steps) / max(len(steps), 1),
    "episodes": eps,
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(f"{pathlib.Path(sys.argv[1]).stem}: {s}/{n} = {p:.1%} ± {out['stderr']:.1%}")
