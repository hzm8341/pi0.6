from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from openpi.training.recap_episode_io import ReCAPFrame, ReCAPOfflineEpisode, save_recap_episodes


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _episode_chunk(episode_index: int, chunks_size: int | None) -> int:
    if not chunks_size:
        return 0
    return episode_index // chunks_size


def _format_dataset_path(template: str, *, episode_index: int, episode_chunk: int, video_key: str | None = None) -> str:
    values: dict[str, Any] = {
        "episode_index": episode_index,
        "episode_chunk": episode_chunk,
    }
    if video_key is not None:
        values["video_key"] = video_key
    return template.format(**values)


def _scalar(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _array(value: Any) -> np.ndarray:
    return np.asarray(_scalar(value), dtype=np.float32)


def _task_lookup(dataset_root: Path) -> dict[int, str]:
    tasks = {}
    for row in _read_jsonl(dataset_root / "meta" / "tasks.jsonl"):
        if "task_index" in row and "task" in row:
            tasks[int(row["task_index"])] = str(row["task"])
    return tasks


def _episode_task(episode_row: Mapping[str, Any], tasks_by_index: Mapping[int, str], columns: Mapping[str, list[Any]]) -> str:
    tasks = episode_row.get("tasks")
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    if "task" in episode_row:
        return str(episode_row["task"])
    task_indices = columns.get("task_index", [])
    if task_indices:
        task_index = int(_scalar(task_indices[0]))
        return tasks_by_index.get(task_index, "")
    return ""


def _load_success_labels(path: str | Path | None) -> dict[str, bool]:
    if path is None:
        return {}
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(key): bool(value) for key, value in data.items()}
        if isinstance(data, list):
            return _success_labels_from_rows(data)
    if path.suffix == ".jsonl":
        return _success_labels_from_rows(_read_jsonl(path))
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as file:
            return _success_labels_from_rows(list(csv.DictReader(file)))
    raise ValueError(f"Unsupported success label file: {path}")


def _success_labels_from_rows(rows: list[Mapping[str, Any]]) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    for row in rows:
        if "success" not in row:
            continue
        key = row.get("episode_id", row.get("episode_index"))
        if key is None:
            continue
        if isinstance(key, int):
            labels[f"episode_{key:06d}"] = bool(row["success"])
        labels[str(key)] = bool(row["success"])
    return labels


def _video_paths(info: Mapping[str, Any], dataset_root: Path, episode_index: int, episode_chunk: int) -> dict[str, str]:
    video_template = info.get("video_path")
    if not video_template:
        return {}
    paths = {}
    for key, feature in info.get("features", {}).items():
        if isinstance(feature, Mapping) and feature.get("dtype") == "video":
            paths[key] = str(
                dataset_root
                / _format_dataset_path(
                    str(video_template),
                    episode_index=episode_index,
                    episode_chunk=episode_chunk,
                    video_key=key,
                )
            )
    return paths


def convert_lerobot_dataset_to_recap(
    lerobot_root: str | Path,
    output_episodes: str | Path,
    *,
    default_success: bool | None = None,
    success_labels_path: str | Path | None = None,
    max_episodes: int | None = None,
) -> list[ReCAPOfflineEpisode]:
    """Convert a local LeRobot v2-style dataset directory to RECAP episode JSON files.

    Unknown success labels are written as ``success=False`` plus
    ``metadata.success_needs_review=True`` so the offline labeling step can run
    but users can still find episodes that need manual success/failure review.
    """
    lerobot_root = Path(lerobot_root)
    info_path = lerobot_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot metadata file: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))

    data_template = str(info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"))
    chunks_size = info.get("chunks_size")
    chunks_size = int(chunks_size) if chunks_size else None
    tasks_by_index = _task_lookup(lerobot_root)
    success_labels = _load_success_labels(success_labels_path)

    episode_rows = _read_jsonl(lerobot_root / "meta" / "episodes.jsonl")
    if not episode_rows:
        raise FileNotFoundError(f"Missing or empty LeRobot episodes metadata: {lerobot_root / 'meta' / 'episodes.jsonl'}")

    episodes: list[ReCAPOfflineEpisode] = []
    for episode_row in episode_rows[:max_episodes]:
        episode_index = int(episode_row["episode_index"])
        episode_id = f"episode_{episode_index:06d}"
        episode_chunk = _episode_chunk(episode_index, chunks_size)
        parquet_path = lerobot_root / _format_dataset_path(
            data_template,
            episode_index=episode_index,
            episode_chunk=episode_chunk,
        )
        if not parquet_path.exists():
            raise FileNotFoundError(f"Missing LeRobot episode parquet: {parquet_path}")

        table = pq.read_table(parquet_path)
        columns = {name: table[name].to_pylist() for name in table.column_names}
        if "action" not in columns:
            raise ValueError(f"LeRobot episode {parquet_path} does not contain an action column")

        frames: list[ReCAPFrame] = []
        frame_count = len(columns["action"])
        for row_idx in range(frame_count):
            observation: dict[str, Any] = {}
            if "observation.state" in columns:
                observation["state"] = _array(columns["observation.state"][row_idx])
            if "timestamp" in columns:
                observation["timestamp"] = float(_scalar(columns["timestamp"][row_idx]))
            if "frame_index" in columns:
                observation["frame_index"] = int(_scalar(columns["frame_index"][row_idx]))
            if "episode_index" in columns:
                observation["episode_index"] = int(_scalar(columns["episode_index"][row_idx]))
            if "task_index" in columns:
                observation["task_index"] = int(_scalar(columns["task_index"][row_idx]))
            if "index" in columns:
                observation["index"] = int(_scalar(columns["index"][row_idx]))

            frames.append(
                ReCAPFrame(
                    t=int(_scalar(columns.get("frame_index", [row_idx] * frame_count)[row_idx])),
                    observation=observation,
                    action=_array(columns["action"][row_idx]),
                    is_human_intervention=False,
                )
            )

        has_label = episode_id in success_labels or str(episode_index) in success_labels
        success = success_labels.get(episode_id, success_labels.get(str(episode_index), default_success))
        metadata: dict[str, Any] = {
            "source": "lerobot",
            "lerobot_root": str(lerobot_root),
            "episode_index": episode_index,
            "length": frame_count,
            "fps": info.get("fps"),
            "success_needs_review": success is None and not has_label,
            "video_paths": _video_paths(info, lerobot_root, episode_index, episode_chunk),
        }

        episodes.append(
            ReCAPOfflineEpisode(
                episode_id=episode_id,
                task=_episode_task(episode_row, tasks_by_index, columns),
                success=bool(success) if success is not None else False,
                timeout=bool(episode_row.get("timeout", False)),
                max_episode_length=episode_row.get("length"),
                frames=frames,
                metadata=metadata,
            )
        )

    save_recap_episodes(episodes, output_episodes)
    return episodes
