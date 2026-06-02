from __future__ import annotations

import argparse
import logging
from pathlib import Path

from openpi.training.recap_episode_io import load_recap_episodes
from openpi.training.recap_offline import build_offline_recap_labels, write_recap_label_outputs


logger = logging.getLogger("recap.label")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate offline RECAP advantage labels from episode JSON files."
    )
    parser.add_argument(
        "--input-episodes",
        type=Path,
        required=True,
        help="Episode JSON file or directory of *.json episodes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for recap_labels.jsonl and lerobot_fields.npz.",
    )
    parser.add_argument(
        "--positive-fraction",
        type=float,
        default=0.4,
        help="Target fraction of positive advantage samples.",
    )
    parser.add_argument(
        "--n-step-lookahead",
        type=int,
        default=50,
        help="N-step lookahead used in advantage estimation.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    episodes = load_recap_episodes(args.input_episodes)
    logger.info("Loaded %d episode(s) from %s", len(episodes), args.input_episodes)
    result = build_offline_recap_labels(
        episodes,
        positive_fraction=args.positive_fraction,
        n_step_lookahead=args.n_step_lookahead,
    )
    write_recap_label_outputs(result, args.output_dir)
    logger.info("Wrote %d labeled timestep(s) to %s", len(result.records), args.output_dir)


if __name__ == "__main__":
    main()
