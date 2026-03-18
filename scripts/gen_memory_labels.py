#!/usr/bin/env python3
"""Offline language-memory label generation script for π0.6-MEM.

Reads a robot dataset whose episodes contain subtask annotations and calls an
LLM (Anthropic Claude by default) to produce compressed memory summaries
``m_t`` at every step.  The resulting JSON-Lines file is used as training
supervision for the high-level policy's language-memory mechanism.

Usage
-----
    python scripts/gen_memory_labels.py \\
        --dataset_path /path/to/dataset \\
        --output_path  /path/to/memory_labels.jsonl \\
        [--api_key     YOUR_ANTHROPIC_API_KEY] \\
        [--max_episodes 100]

Input format (``subtask_annotations.jsonl`` inside ``dataset_path``)
----------------------------------------------------------------------
Each line is a JSON object::

    {
        "episode_id": "ep0001",
        "subtasks": [
            {"instruction": "Move pot to sink",       "success": true},
            {"instruction": "Grab potatoes from fridge", "success": true},
            {"instruction": "Grab milk – wrong shelf",   "success": false},
            {"instruction": "Grab milk from fridge",  "success": true}
        ]
    }

Output format (``memory_labels.jsonl``)
-----------------------------------------
One line per subtask step::

    {
        "episode_id": "ep0001",
        "timestep":   0,
        "subtask_instruction": "Move pot to sink",
        "subtask_success":     true,
        "memory_before":       "",
        "memory_after":        "I moved the pot to the sink."
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM client wrapper
# ---------------------------------------------------------------------------

def _build_anthropic_client(api_key: str):
    """Return a thin wrapper around the Anthropic SDK."""
    try:
        import anthropic
    except ImportError:
        logger.error("Package 'anthropic' not found. Run: pip install anthropic")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    class _Client:
        def generate(self, prompt: str, *, temperature: float = 0.3, max_tokens: int = 256) -> str:
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text

    return _Client()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_episodes(dataset_path: str) -> list[dict]:
    """Load episode subtask annotations from ``<dataset_path>/subtask_annotations.jsonl``."""
    ann_file = os.path.join(dataset_path, "subtask_annotations.jsonl")
    if not os.path.exists(ann_file):
        logger.error("Annotation file not found: %s", ann_file)
        sys.exit(1)

    episodes: list[dict] = []
    with open(ann_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    logger.info("Loaded %d episodes from %s", len(episodes), ann_file)
    return episodes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate language memory training labels.")
    parser.add_argument("--dataset_path",  required=True,  help="Path to robot dataset directory.")
    parser.add_argument("--output_path",   required=True,  help="Output JSON-Lines path.")
    parser.add_argument("--api_key",       default=os.environ.get("ANTHROPIC_API_KEY"),
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var).")
    parser.add_argument("--max_episodes",  type=int, default=None,
                        help="Process only the first N episodes (for debugging).")
    parser.add_argument("--max_memory_len", type=int, default=512,
                        help="Maximum characters per memory summary.")
    args = parser.parse_args()

    if not args.api_key:
        logger.error("Anthropic API key is required.  Pass --api_key or set ANTHROPIC_API_KEY.")
        sys.exit(1)

    # Lazy import so the script can be imported without the openpi package
    # being fully installed.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from openpi.models.memory_manager import MemoryDataGenerator, MemoryGenerationConfig

    llm_client = _build_anthropic_client(args.api_key)
    generator = MemoryDataGenerator(
        llm_client=llm_client,
        config=MemoryGenerationConfig(max_memory_length=args.max_memory_len),
    )

    episodes = load_episodes(args.dataset_path)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    all_labels = []
    for i, ep in enumerate(episodes):
        ep_labels = generator.generate_labels_for_episode(
            episode_id=ep["episode_id"],
            subtasks=ep["subtasks"],
        )
        all_labels.extend(ep_labels)
        if (i + 1) % 10 == 0 or (i + 1) == len(episodes):
            logger.info("Progress: %d / %d episodes, %d labels total",
                        i + 1, len(episodes), len(all_labels))

    generator.save_labels(all_labels, args.output_path)
    logger.info("Done.  Wrote %d labels to %s", len(all_labels), args.output_path)


if __name__ == "__main__":
    main()
