"""Strict recorded-episode input and train-only normalization.

Episodes contain synchronized observations and actions in the declared
coordinate frames and units. Calibration and action conversion are performed
during data preparation. See docs/DATA_FORMAT.md for the acquisition format.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


FINGER_ORDER = ["thumb", "index", "middle", "ring", "little"]
_EXPECTED_METADATA = {
    "finger_order": FINGER_ORDER,
    "position_frame": "wrist", "position_unit": "m",
    "force_frame": "sensor_local", "force_unit": "N",
    "force_axes": "common_calibrated",
    "action_convention": "arm_delta_hand_target", "timestamp_unit": "s",
}
_PROVENANCE_FIELDS = (
    "arm_translation_frame", "arm_rotation_representation",
    "arm_rotation_composition", "hand_command_unit", "position_source",
    "force_calibration_id",
)
_ARRAY_KEYS = (
    "images", "proprio", "positions", "forces", "actions",
    "timestamps", "image_timestamps",
)


def _episode_path(root: Path, filename: str) -> Path:
    if not isinstance(filename, str) or not filename:
        raise ValueError("Episode file must be a nonempty relative NPZ path")
    relative = Path(filename)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Episode file must remain inside the dataset root")
    if path.suffix != ".npz" or not path.is_file():
        raise ValueError(f"Episode file is not an existing .npz file: {filename}")
    return path


def _validate_metadata(metadata: dict) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("manifest requires explicit physical metadata")
    for key, expected in _EXPECTED_METADATA.items():
        if metadata.get(key) != expected:
            raise ValueError(f"metadata.{key} must be {expected!r}; no implicit conversion is performed")
    for key in _PROVENANCE_FIELDS:
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"metadata.{key} must be a nonempty recorded declaration")
    for key in ("action_dim", "proprio_dim", "num_cameras"):
        if type(metadata.get(key)) is not int or metadata[key] <= 0:
            raise ValueError(f"metadata.{key} must be a positive integer")
    if metadata["action_dim"] not in (26, 28):
        raise ValueError("metadata.action_dim must be exactly 26 or 28")


def read_manifest(root: str | Path) -> dict[str, Any]:
    """Read and validate schema/provenance without loading episode arrays."""
    root = Path(root)
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    _validate_metadata(manifest.get("metadata"))
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("manifest episodes must be a nonempty list")
    ids, paths = set(), set()
    for episode in episodes:
        if not isinstance(episode, dict) or not isinstance(episode.get("id"), str) or not episode["id"]:
            raise ValueError("Each episode requires a nonempty string id")
        if episode["id"] in ids:
            raise ValueError(f"duplicate episode id: {episode['id']}")
        path = _episode_path(root, episode.get("file"))
        if path in paths:
            raise ValueError("duplicate episode file would invalidate episode splitting")
        ids.add(episode["id"])
        paths.add(path)
    return manifest


def _select_episodes(manifest: dict, episode_ids: Sequence[str]) -> list[dict]:
    ids = list(episode_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("episode_ids must be nonempty and unique")
    entries = {entry["id"]: entry for entry in manifest["episodes"]}
    if any(episode_id not in entries for episode_id in ids):
        raise ValueError("episode_ids contains an unknown episode")
    return [entries[episode_id] for episode_id in ids]


def _load_episode(root: Path, entry: dict, metadata: dict) -> dict[str, np.ndarray]:
    with np.load(_episode_path(root, entry["file"]), allow_pickle=False) as source:
        missing = set(_ARRAY_KEYS) - set(source.files)
        if missing:
            raise ValueError(f"Episode {entry['id']} missing required fields: {sorted(missing)}")
        arrays = {key: source[key] for key in _ARRAY_KEYS}
    times = arrays["timestamps"]
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("timestamps must have nonempty shape [N]")
    n, v = len(times), metadata["num_cameras"]
    shapes = {
        "proprio": (n, metadata["proprio_dim"]),
        "actions": (n, metadata["action_dim"]),
        "positions": (n, 5, 3), "forces": (n, 5, 3),
        "timestamps": (n,), "image_timestamps": (n, v),
    }
    for key, expected in shapes.items():
        value = arrays[key]
        if value.shape != expected:
            raise ValueError(f"{entry['id']}/{key} shape {value.shape}; expected {expected}")
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"{entry['id']}/{key} must contain finite numeric values")
    if times.dtype.kind != "f" or arrays["image_timestamps"].dtype.kind != "f":
        raise ValueError("timestamps and image_timestamps must be floating seconds")
    if np.any(np.diff(times.astype(np.float64)) <= 0):
        raise ValueError("timestamps must be strictly increasing within an episode")
    image_times = arrays["image_timestamps"].astype(np.float64)
    if np.any(np.diff(image_times, axis=0) < 0):
        raise ValueError("image_timestamps must be nondecreasing per camera")
    if np.any(image_times > times.astype(np.float64)[:, None]):
        raise ValueError("image_timestamps contains future images unavailable at the stored observation time")
    images = arrays["images"]
    if images.ndim != 5 or images.shape[:3] != (n, v, 3) or min(images.shape[3:]) < 1:
        raise ValueError("images must have shape [N,V,3,H,W]")
    if images.dtype != np.uint8:
        if images.dtype.kind != "f" or not np.isfinite(images).all() or np.min(images) < 0 or np.max(images) > 1:
            raise ValueError("images must be uint8 [0,255] or floating [0,1]")
    return arrays


def split_episode_ids(root: str | Path, val_fraction: float = 0.2, seed: int = 0) -> tuple[list[str], list[str]]:
    """Deterministic episode split; one episode remains train-only.

    A zero validation fraction explicitly requests train-only operation.
    For positive fractions and two or more episodes both sets are nonempty.
    """
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must lie in [0,1)")
    ids = [entry["id"] for entry in read_manifest(root)["episodes"]]
    if len(ids) == 1 or val_fraction == 0:
        return ids, []
    n_val = min(len(ids) - 1, max(1, round(len(ids) * val_fraction)))
    chosen = set(np.random.default_rng(seed).choice(len(ids), n_val, replace=False).tolist())
    return ([value for i, value in enumerate(ids) if i not in chosen],
            [value for i, value in enumerate(ids) if i in chosen])


class Normalizer:
    """Train-only scalers; vectors use one scalar and zero spatial offset."""

    def __init__(self, state: dict[str, Any]):
        self._state = copy.deepcopy(state)

    @classmethod
    def fit(cls, root: str | Path, train_ids: Sequence[str]) -> "Normalizer":
        root = Path(root)
        manifest = read_manifest(root)
        entries = _select_episodes(manifest, train_ids)
        moments: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
        maxima = {"positions": 0.0, "forces": 0.0}
        for entry in entries:
            arrays = _load_episode(root, entry, manifest["metadata"])
            for key in maxima:
                maxima[key] = max(maxima[key], float(np.linalg.norm(arrays[key].astype(np.float64), axis=-1).max()))
            for key in ("actions", "proprio"):
                values = arrays[key].astype(np.float64)
                count, mean = len(values), values.mean(axis=0)
                m2 = np.sum((values - mean) ** 2, axis=0)
                if key in moments:
                    previous_count, previous_mean, previous_m2 = moments[key]
                    difference = mean - previous_mean
                    total = count + previous_count
                    m2 += previous_m2 + difference ** 2 * previous_count * count / total
                    mean = previous_mean + difference * count / total
                    count = total
                moments[key] = (count, mean, m2)
        state: dict[str, Any] = {
            "schema_version": 1, "train_episode_ids": [entry["id"] for entry in entries],
            "position_scale": 1.0 / max(maxima["positions"], 1e-8),
            "force_scale": 1.0 / max(maxima["forces"], 1e-8),
            "metadata": copy.deepcopy(manifest["metadata"]),
        }
        for key, (count, mean, m2) in moments.items():
            std = np.sqrt(np.maximum(m2 / count, 0))
            state[key + "_mean"] = mean.tolist()
            state[key + "_std"] = np.where(std > 1e-6, std, 1.0).tolist()
        return cls.from_state_dict(state)

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "Normalizer":
        if state.get("schema_version") != 1 or not state.get("train_episode_ids"):
            raise ValueError("Normalizer requires version 1 and recorded train_episode_ids")
        for key in ("position_scale", "force_scale"):
            if not np.isscalar(state.get(key)) or not np.isfinite(state[key]) or state[key] <= 0:
                raise ValueError(f"Normalizer {key} must be a positive finite scalar")
        for key in ("actions", "proprio"):
            mean, std = np.asarray(state.get(key + "_mean")), np.asarray(state.get(key + "_std"))
            if mean.ndim != 1 or std.shape != mean.shape or not mean.size:
                raise ValueError(f"Normalizer requires matching {key} mean/std vectors")
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError(f"Normalizer {key} statistics must be finite with positive std")
        return cls(state)

    @staticmethod
    def _require_floating(values):
        floating = (torch.is_floating_point(values) if isinstance(values, torch.Tensor)
                    else isinstance(values, np.ndarray) and values.dtype.kind == "f")
        if not floating:
            raise ValueError("Normalizer inputs must be floating NumPy arrays or Torch tensors")

    @staticmethod
    def _parameter(value: Any, reference: np.ndarray | torch.Tensor):
        Normalizer._require_floating(reference)
        if isinstance(reference, torch.Tensor):
            return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        return np.asarray(value, dtype=np.asarray(reference).dtype)

    def normalize_tactile(self, positions, forces):
        self._require_floating(positions)
        self._require_floating(forces)
        return (positions * self._state["position_scale"],
                forces * self._state["force_scale"])

    def normalize_proprio(self, values):
        return ((values - self._parameter(self._state["proprio_mean"], values)) /
                self._parameter(self._state["proprio_std"], values))

    def denormalize_actions(self, values):
        return (values * self._parameter(self._state["actions_std"], values) +
                self._parameter(self._state["actions_mean"], values))

    def normalize_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        result = dict(batch)
        for key, scale in (("positions", "position_scale"), ("forces", "force_scale"), ("future_forces", "force_scale")):
            if key in result:
                self._require_floating(result[key])
                result[key] = result[key] * self._state[scale]
        if "proprio" in result:
            result["proprio"] = self.normalize_proprio(result["proprio"])
        if "actions" in result:
            values = result["actions"]
            result["actions"] = ((values - self._parameter(self._state["actions_mean"], values)) /
                                 self._parameter(self._state["actions_std"], values))
        return result


class EpisodeDataset(Dataset):
    """Full valid within-episode windows, with frozen causal slow context.

    NPZ decoding is episode-granular: validation processes one episode at a
    time, and each worker caches at most one decoded episode during sampling.
    It does not copy every episode's images into RAM simultaneously.
    """

    def __init__(self, root: str | Path, episode_ids: Sequence[str], normalizer: Normalizer | None = None,
                 observation_horizon: int = 2, action_horizon: int = 16,
                 prediction_horizon: int = 8, slow_stride: int = 16, sample_stride: int = 1):
        self.root = Path(root)
        self.manifest = read_manifest(root)
        entries = _select_episodes(self.manifest, episode_ids)
        for name, value in (("observation_horizon", observation_horizon), ("action_horizon", action_horizon),
                            ("prediction_horizon", prediction_horizon), ("slow_stride", slow_stride), ("sample_stride", sample_stride)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.normalizer = normalizer
        if normalizer is not None:
            state = normalizer.state_dict()
            if state.get("metadata") != self.manifest["metadata"]:
                raise ValueError("Normalizer metadata differs from dataset metadata; use statistics for the declared physical contract")
            if len(state["actions_mean"]) != self.manifest["metadata"]["action_dim"] or len(state["proprio_mean"]) != self.manifest["metadata"]["proprio_dim"]:
                raise ValueError("Normalizer feature dimensions do not match the dataset")
        self.observation_horizon = observation_horizon
        self.action_horizon = action_horizon
        self.prediction_horizon = prediction_horizon
        self.slow_stride = slow_stride
        self.episode_ids = [entry["id"] for entry in entries]
        self.indices: list[tuple[int, int]] = []
        self._entries = entries
        self._cached_index: int | None = None
        self._cached_episode: dict | None = None
        for index, entry in enumerate(entries):
            episode = _load_episode(self.root, entry, self.manifest["metadata"])
            first = (observation_horizon - 1) * slow_stride
            last = len(episode["timestamps"]) - action_horizon - prediction_horizon
            self.indices.extend((index, start) for start in range(first, last + 1, sample_stride))
        if not self.indices:
            raise ValueError("No full valid windows: episodes are too short for slow context and future targets")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, start = self.indices[index]
        if episode_index != self._cached_index:
            self._cached_episode = _load_episode(self.root, self._entries[episode_index], self.manifest["metadata"])
            self._cached_index = episode_index
        arrays = self._cached_episode
        slow = start - np.arange(self.observation_horizon - 1, -1, -1) * self.slow_stride
        stop = start + self.action_horizon
        image_values = arrays["images"][slow]
        images = image_values.astype(np.float32)
        if image_values.dtype == np.uint8:
            images /= 255.0
        result = {
            "images": torch.from_numpy(images),
            "proprio": torch.from_numpy(arrays["proprio"][slow].astype(np.float32)),
            "positions": torch.from_numpy(arrays["positions"][start:stop].astype(np.float32)),
            "forces": torch.from_numpy(arrays["forces"][start:stop].astype(np.float32)),
            "future_forces": torch.from_numpy(arrays["forces"][stop:stop + self.prediction_horizon].astype(np.float32)),
            "actions": torch.from_numpy(arrays["actions"][start:stop].astype(np.float32)),
        }
        return result if self.normalizer is None else self.normalizer.normalize_batch(result)
