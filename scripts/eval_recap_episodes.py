from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpi.training.recap_dataset_tools import summarize_episodes
from openpi.training.recap_episode_io import load_recap_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RECAP episode JSON data before training.")
    parser.add_argument("--input-episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-action-threshold", type=float, default=0.05)
    args = parser.parse_args()

    episodes = load_recap_episodes(args.input_episodes)
    summary = summarize_episodes(episodes, static_action_threshold=args.static_action_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
