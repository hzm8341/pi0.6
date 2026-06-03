from __future__ import annotations

import argparse
from pathlib import Path

from openpi.training.recap_dataset_tools import trim_tail_static_frames
from openpi.training.recap_episode_io import load_recap_episodes, save_recap_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim static tail frames from RECAP episode JSON data.")
    parser.add_argument("--input-episodes", type=Path, required=True)
    parser.add_argument("--output-episodes", type=Path, required=True)
    parser.add_argument("--action-norm-threshold", type=float, default=0.05)
    parser.add_argument("--min-frames", type=int, default=10)
    args = parser.parse_args()

    episodes = load_recap_episodes(args.input_episodes)
    trimmed = trim_tail_static_frames(
        episodes,
        action_norm_threshold=args.action_norm_threshold,
        min_frames=args.min_frames,
    )
    save_recap_episodes(trimmed, args.output_episodes)


if __name__ == "__main__":
    main()
