from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from openpi.training.recap_config import ReCAPTrainConfig


logger = logging.getLogger("recap")


def run_recap(config: ReCAPTrainConfig) -> None:
    logger.info("Starting RECAP task %s", config.task_name)
    logger.info("Demo dataset: %s", config.demo_dataset_path)
    logger.info("Output directory: %s", config.output_dir)

    for iteration in range(config.num_iterations):
        iter_dir = config.output_dir / f"iter_{iteration:03d}"
        logger.info("[%03d] load demos and collected episodes", iteration)
        logger.info("[%03d] train value function for %d steps", iteration, config.iteration.value_train_steps)
        logger.info("[%03d] assign advantage labels with positive fraction %.2f", iteration, config.iteration.positive_fraction)
        logger.info("[%03d] fine-tune VLA from base checkpoint for %d steps", iteration, config.iteration.vla_train_steps)
        logger.info("[%03d] collect %d rollout episodes into %s", iteration, config.iteration.collect_episodes, iter_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="RECAP iterative training")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--demo-dataset", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/recap"))
    parser.add_argument("--num-iterations", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    config = ReCAPTrainConfig(
        task_name=args.task_name,
        demo_dataset_path=args.demo_dataset,
        output_dir=args.output_dir,
        num_iterations=args.num_iterations,
    )
    logger.info("Loaded config: %s", dataclasses.asdict(config))
    run_recap(config)


if __name__ == "__main__":
    main()
