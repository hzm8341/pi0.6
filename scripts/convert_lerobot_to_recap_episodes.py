#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from openpi.training.lerobot_to_recap import convert_lerobot_dataset_to_recap


def _parse_default_success(value: str) -> bool | None:
    normalized = value.lower()
    if normalized in {"unknown", "none"}:
        return None
    if normalized in {"true", "success", "1", "yes"}:
        return True
    if normalized in {"false", "failure", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Use one of: unknown, true, false")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a local LeRobot dataset to RECAP episode JSON files.")
    parser.add_argument("--lerobot-root", type=Path, required=True, help="Local LeRobot dataset directory.")
    parser.add_argument("--output-episodes", type=Path, required=True, help="Output RECAP episode JSON directory or .json file.")
    parser.add_argument(
        "--default-success",
        type=_parse_default_success,
        default=None,
        help="Default success label for all episodes: unknown, true, or false. Default: unknown.",
    )
    parser.add_argument(
        "--success-labels",
        type=Path,
        default=None,
        help="Optional JSON/JSONL/CSV labels with episode_id or episode_index plus success.",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional limit for smoke conversion.")
    args = parser.parse_args()

    episodes = convert_lerobot_dataset_to_recap(
        args.lerobot_root,
        args.output_episodes,
        default_success=args.default_success,
        success_labels_path=args.success_labels,
        max_episodes=args.max_episodes,
    )
    review_count = sum(episode.metadata.get("success_needs_review", False) for episode in episodes)
    print(f"Wrote {len(episodes)} RECAP episode(s) to {args.output_episodes}")
    if review_count:
        print(f"{review_count} episode(s) have unknown success labels and need manual review.")


if __name__ == "__main__":
    main()
