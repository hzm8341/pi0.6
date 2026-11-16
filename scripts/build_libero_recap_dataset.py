"""Build a local LeRobot dataset (and matching RECAP episode JSON) for the LIBERO RECAP experiment.

Sources:
  * Demonstrations: episodes downloaded from `physical-intelligence/libero` (LeRobot v2.0 parquet files with
    embedded PNG images), all treated as successful.
  * Rollouts: npz files written by `examples/libero/main.py --record-dir` (images already 256x256 and rotated
    180 degrees, exactly like the demos).

The LeRobot dataset and `recap_episodes.json` contain the same episodes in the same frame order, so the sidecar
produced by `scripts/label_recap_advantage.py --input-episodes <out>/recap_episodes.json` lines up with the
dataset index. See docs/recap_libero.md.

Examples:
  # Few-shot SFT dataset with 10 demos
  python scripts/build_libero_recap_dataset.py --repo-id local/libero_task0_demos10 --num-demos 10
  # All demos (used for the shared norm stats)
  python scripts/build_libero_recap_dataset.py --repo-id local/libero_task0_demos33
  # RECAP dataset: the same 10 demos + recorded rollouts
  python scripts/build_libero_recap_dataset.py --repo-id local/libero_task0_recap --num-demos 10 \\
      --rollout-dir data/libero/rollouts_sft
"""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import pathlib
import shutil

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import tyro

TASK0 = "put both the alphabet soup and the tomato sauce in the basket"
# Episode step limit used by examples/libero/main.py for libero_10; normalizes rewards/returns to (-1, 0).
LIBERO_10_MAX_STEPS = 520

FEATURES = {
    "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}


@dataclasses.dataclass
class Args:
    repo_id: str
    """Output LeRobot repo id (stored under $HF_LEROBOT_HOME)."""
    demo_dir: str = "/media/hzm/data24t/datasets/libero_src"
    """Local copy of physical-intelligence/libero (meta/ + the downloaded parquet files)."""
    task: str = TASK0
    num_demos: int | None = None
    """Use this many demos (a seeded random subset); None = all demos of the task."""
    demo_seed: int = 0
    rollout_dir: str | None = None
    """Directory of rollout npz files from examples/libero/main.py --record-dir."""
    max_rollouts: int | None = None
    max_episode_length: int = LIBERO_10_MAX_STEPS
    demos_as_intervention: bool = False
    """Mark demo frames as human interventions, which forces them positive during advantage labeling."""
    overwrite: bool = False


@dataclasses.dataclass
class _Episode:
    episode_id: str
    source: str
    success: bool
    image: np.ndarray  # (T, 256, 256, 3) uint8
    wrist_image: np.ndarray
    state: np.ndarray  # (T, 8)
    actions: np.ndarray  # (T, 7)


def _decode(cell: dict) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


def select_demo_episodes(demo_dir: pathlib.Path, task: str, num_demos: int | None, seed: int) -> list[int]:
    with open(demo_dir / "meta" / "episodes.jsonl") as f:
        episodes = [json.loads(line) for line in f]
    selected = sorted(e["episode_index"] for e in episodes if e["tasks"] == [task])
    if not selected:
        raise ValueError(f"No demos for task {task!r} in {demo_dir}")
    if num_demos is not None:
        if num_demos > len(selected):
            raise ValueError(f"Requested {num_demos} demos, only {len(selected)} available")
        rng = np.random.default_rng(seed)
        selected = sorted(rng.choice(selected, size=num_demos, replace=False).tolist())
    return selected


def load_demo(demo_dir: pathlib.Path, info: dict, episode_index: int) -> _Episode:
    rel = info["data_path"].format(episode_chunk=episode_index // info["chunks_size"], episode_index=episode_index)
    table = pq.read_table(demo_dir / rel).to_pydict()
    order = np.argsort(table["frame_index"])
    return _Episode(
        episode_id=f"demo_{episode_index:06d}",
        source="demo",
        success=True,
        image=np.stack([_decode(table["image"][i]) for i in order]),
        wrist_image=np.stack([_decode(table["wrist_image"][i]) for i in order]),
        state=np.asarray(table["state"], dtype=np.float32)[order],
        actions=np.asarray(table["actions"], dtype=np.float32)[order],
    )


def load_rollout(path: pathlib.Path) -> _Episode:
    with np.load(path) as data:
        return _Episode(
            episode_id=f"rollout_{path.stem}",
            source="rollout",
            success=bool(data["success"]),
            image=data["image"],
            wrist_image=data["wrist_image"],
            state=data["state"].astype(np.float32),
            actions=data["actions"].astype(np.float32),
        )


def recap_episode_dict(ep: _Episode, *, task: str, max_episode_length: int, intervention: bool) -> dict:
    return {
        "episode_id": ep.episode_id,
        "task": task,
        "success": ep.success,
        "max_episode_length": max_episode_length,
        "frames": [
            {"t": t, "action": ep.actions[t].tolist(), "is_human_intervention": intervention}
            for t in range(len(ep.actions))
        ],
        "metadata": {"source": ep.source},
    }


def main(args: Args) -> None:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite")
        shutil.rmtree(output_path)

    demo_dir = pathlib.Path(args.demo_dir)
    info = json.loads((demo_dir / "meta" / "info.json").read_text())
    demo_ids = select_demo_episodes(demo_dir, args.task, args.num_demos, args.demo_seed)
    rollout_paths = sorted(pathlib.Path(args.rollout_dir).glob("*.npz")) if args.rollout_dir else []
    if args.max_rollouts is not None:
        rollout_paths = rollout_paths[: args.max_rollouts]
    logging.info("Building %s from %d demos and %d rollouts", args.repo_id, len(demo_ids), len(rollout_paths))

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=10,
        features=FEATURES,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    recap_episodes = []
    sources = [("demo", i) for i in demo_ids] + [("rollout", p) for p in rollout_paths]
    for n, (kind, ref) in enumerate(sources):
        ep = load_demo(demo_dir, info, ref) if kind == "demo" else load_rollout(ref)
        num_frames = len(ep.actions)
        if not (len(ep.image) == len(ep.wrist_image) == len(ep.state) == num_frames) or num_frames == 0:
            raise ValueError(f"{ep.episode_id}: inconsistent or empty episode")
        for t in range(num_frames):
            dataset.add_frame(
                {
                    "image": ep.image[t],
                    "wrist_image": ep.wrist_image[t],
                    "state": ep.state[t],
                    "actions": ep.actions[t],
                    "task": args.task,
                }
            )
        dataset.save_episode()
        recap_episodes.append(
            recap_episode_dict(
                ep,
                task=args.task,
                max_episode_length=args.max_episode_length,
                intervention=args.demos_as_intervention and kind == "demo",
            )
        )
        logging.info("[%d/%d] %s: %d frames, success=%s", n + 1, len(sources), ep.episode_id, num_frames, ep.success)

    # Same episode order as the LeRobot dataset -> sidecar arrays line up with the flattened frame index.
    (output_path / "recap_episodes.json").write_text(json.dumps(recap_episodes))
    summary = {
        "repo_id": args.repo_id,
        "demo_episode_indices": demo_ids,
        "rollouts": [p.name for p in rollout_paths],
        "num_episodes": len(recap_episodes),
        "num_frames": int(sum(len(e["frames"]) for e in recap_episodes)),
        "num_successful_rollouts": int(
            sum(e["success"] for e in recap_episodes if e["metadata"]["source"] == "rollout")
        ),
        "args": dataclasses.asdict(args),
    }
    (output_path / "recap_build_summary.json").write_text(json.dumps(summary, indent=2))
    logging.info("Done: %s", json.dumps({k: v for k, v in summary.items() if k not in {"args", "rollouts"}}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    main(tyro.cli(Args))
