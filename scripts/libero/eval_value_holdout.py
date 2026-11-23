"""Value-model and advantage-label quality on held-out episodes, for several advantage estimators.

Usage: python scripts/libero/eval_value_holdout.py <labels_dir_with_values> <holdout_episodes.json> [--out x.json]
<labels_dir_with_values> is the output of label_recap_with_value.py (recap_labels.jsonl rows carry `value`).

Value metrics (train vs held-out frames): Spearman(V, return), MAE, and AUC of V(o_0) for episode success.
Advantage variants, each thresholded at the global top-30% (as in training): N-step with N in {10, 50, 100, 200},
Monte Carlo (A = G_t - V_t), and the success-gated N=50 labels (RECAP r2 recipe). For each we report, on held-out
rollouts, the positive rate in successful / failed episodes, the share of positives from failed episodes and the
frame-level AUC of A for episode success. These are proxies: per-frame ground truth does not exist.
"""

import argparse
import json
import pathlib

import numpy as np
from scipy.stats import rankdata
from scipy.stats import spearmanr

from openpi.training.recap_value_proxy import compute_n_step_advantages

ap = argparse.ArgumentParser()
ap.add_argument("labels_dir", type=pathlib.Path)
ap.add_argument("holdout", type=pathlib.Path)
ap.add_argument("--positive-fraction", type=float, default=0.3)
ap.add_argument("--out", type=pathlib.Path)
a = ap.parse_args()


def auc(score, label):
    label = np.asarray(label, bool)
    r = rankdata(score)
    n1, n0 = label.sum(), (~label).sum()
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


recs = [json.loads(line) for line in (a.labels_dir / "recap_labels.jsonl").open()]
held = set(json.loads(a.holdout.read_text()))
ep = np.asarray([r["episode_id"] for r in recs])
V = np.asarray([r["value"] for r in recs], np.float32)
G = np.asarray([r["return"] for r in recs], np.float32)
R = np.asarray([r["reward"] for r in recs], np.float32)
S = np.asarray([r["success"] for r in recs], bool)
is_held = np.isin(ep, list(held))
is_demo = np.char.startswith(ep.astype(str), "demo")
starts = np.r_[0, np.flatnonzero(ep[1:] != ep[:-1]) + 1]
bounds = list(zip(starts, np.r_[starts[1:], len(ep)], strict=True))

out = {"value": {}}
for name, m in (("train_rollouts", ~is_held & ~is_demo), ("heldout", is_held)):
    first = np.asarray([s for s, _ in bounds if m[s]])
    out["value"][name] = {
        "frames": int(m.sum()),
        "spearman": float(spearmanr(V[m], G[m]).statistic),
        "mae": float(np.abs(V[m] - G[m]).mean()),
        "auc_v0_success": auc(V[first], S[first]),
    }


def nstep(n):
    adv = np.empty_like(V)
    for s, e in bounds:
        adv[s:e] = compute_n_step_advantages(R[s:e], V[s:e], n_step_lookahead=n)
    return adv


variants = {f"nstep_{n}": nstep(n) for n in (10, 50, 100, 200)}
variants["monte_carlo"] = G - V
# Success-gated: failed-episode frames can never be positive (r2 recipe); threshold chosen on the remaining frames.
variants["gated_nstep_50"] = np.where(S, variants["nstep_50"], -np.inf)

out["advantage"] = {}
for name, A in variants.items():
    # Same overall positive count for every variant (gated: all positives come from successful episodes).
    thr = np.sort(A)[-round(a.positive_fraction * len(A))]
    pos = thr <= A
    h = is_held
    out["advantage"][name] = {
        "heldout_pos_rate_succ": float(pos[h & S].mean()),
        "heldout_pos_rate_fail": float(pos[h & ~S].mean()),
        "heldout_fail_share_of_pos": float(pos[h & ~S].sum() / max(pos[h].sum(), 1)),
        "heldout_auc_success": auc(np.where(np.isfinite(A[h]), A[h], -1e9), S[h]),
    }

txt = json.dumps(out, indent=2)
print(txt)
if a.out:
    a.out.write_text(txt)
