from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpi.training.recap_dataset_tools import build_split_manifest
from openpi.training.recap_episode_io import load_recap_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a train/eval split manifest for RECAP episode JSON data.")
    parser.add_argument("--input-episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    episodes = load_recap_episodes(args.input_episodes)
    manifest = build_split_manifest(episodes, eval_fraction=args.eval_fraction, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
