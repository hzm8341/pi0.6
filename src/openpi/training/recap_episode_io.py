from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class ReCAPFrame:
    t: int
    observation: dict[str, Any]
    action: np.ndarray
    is_human_intervention: bool = False


@dataclasses.dataclass(frozen=True)
class ReCAPOfflineEpisode:
    episode_id: str
    task: str
    success: bool
    frames: list[ReCAPFrame]
    timeout: bool = False
    max_episode_length: int | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


def _to_array_if_numeric(value: Any) -> Any:
    if isinstance(value, list) and _is_numeric_list(value):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, dict):
        return {key: _to_array_if_numeric(item) for key, item in value.items()}
    return value


def _is_numeric_list(value: list[Any]) -> bool:
    if not value:
        return False
    return all(isinstance(item, int | float | bool) for item in value)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _episode_from_dict(data: dict[str, Any]) -> ReCAPOfflineEpisode:
    frames = [
        ReCAPFrame(
            t=int(frame.get("t", idx)),
            observation=_to_array_if_numeric(frame.get("observation", frame.get("obs", {}))),
            action=np.asarray(frame["action"], dtype=np.float32),
            is_human_intervention=bool(frame.get("is_human_intervention", False)),
        )
        for idx, frame in enumerate(data["frames"])
    ]
    return ReCAPOfflineEpisode(
        episode_id=str(data["episode_id"]),
        task=str(data.get("task", "")),
        success=bool(data["success"]),
        timeout=bool(data.get("timeout", False)),
        max_episode_length=data.get("max_episode_length"),
        frames=frames,
        metadata=dict(data.get("metadata", {})),
    )


def _episode_to_dict(episode: ReCAPOfflineEpisode) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "task": episode.task,
        "success": episode.success,
        "timeout": episode.timeout,
        "max_episode_length": episode.max_episode_length,
        "frames": [
            {
                "t": frame.t,
                "observation": _to_jsonable(frame.observation),
                "action": _to_jsonable(frame.action),
                "is_human_intervention": frame.is_human_intervention,
            }
            for frame in episode.frames
        ],
        "metadata": _to_jsonable(episode.metadata),
    }


def load_recap_episodes(path: str | Path) -> list[ReCAPOfflineEpisode]:
    path = Path(path)
    if path.is_dir():
        episodes: list[ReCAPOfflineEpisode] = []
        for episode_path in sorted(path.glob("*.json")):
            episodes.extend(load_recap_episodes(episode_path))
        return episodes

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [_episode_from_dict(item) for item in data]
    return [_episode_from_dict(data)]


def save_recap_episodes(episodes: list[ReCAPOfflineEpisode], path: str | Path) -> None:
    path = Path(path)
    if path.suffix == ".json":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([_episode_to_dict(episode) for episode in episodes], indent=2), encoding="utf-8")
        return

    path.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        episode_path = path / f"{episode.episode_id}.json"
        episode_path.write_text(json.dumps(_episode_to_dict(episode), indent=2), encoding="utf-8")
