#!/usr/bin/env python3
"""Convert calibrated replay groups to the Dex-EquiTact episode format.

The optional Zarr dependency is imported by the CLI. The conversion core accepts
an array group, preserves episode boundaries, and validates the recorded fields.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from dex_equitact.data import _load_episode, _validate_metadata, read_manifest


def convert_replay_group(source, output, metadata, *, image_keys=("oak_rgb", "realsense_rgb"),
                         proprio_keys=("tcp_state", "q_state"), timestamp_key="timestamp",
                         image_timestamps_key="image_timestamps", positions_key="tactile_pos",
                         forces_key="tactile", actions_key="action"):
    """Write a new canonical directory after validating every source episode.

    Source image arrays must use NHWC. Sensor fields and commands are copied
    without semantic conversion. The required metadata must describe values
    that already satisfy the canonical physical and action conventions.
    """
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    _validate_metadata(metadata)
    if "data" not in source or "meta" not in source or "episode_ends" not in source["meta"]:
        raise ValueError("Source requires data/* and meta/episode_ends")
    data = source["data"]
    required = list(image_keys) + list(proprio_keys) + [timestamp_key, image_timestamps_key, positions_key, forces_key, actions_key]
    for key in required:
        if key not in data:
            raise ValueError(f"Source missing required recorded field: {key}")
    if len(image_keys) != metadata["num_cameras"] or not proprio_keys:
        raise ValueError("Image/proprio source key counts do not match the declared contract")
    ends = np.asarray(source["meta"]["episode_ends"][:])
    if ends.ndim != 1 or ends.dtype.kind not in "iu" or len(ends) == 0 or np.any(np.diff(ends) <= 0) or ends[0] <= 0:
        raise ValueError("episode_ends must be strictly increasing positive exclusive integer boundaries")
    total = int(ends[-1])
    for key in required:
        if not data[key].shape or data[key].shape[0] != total:
            raise ValueError(f"Source {key} length differs from the final episode boundary")
    if data[timestamp_key].dtype.kind != "f" or data[image_timestamps_key].dtype.kind != "f":
        raise ValueError("timestamp and image_timestamps must already be measured floating seconds")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary)
        (stage / "episodes").mkdir()
        entries = []
        start = 0
        for index, end in enumerate(ends):
            end = int(end)
            images = [np.asarray(data[key][start:end]) for key in image_keys]
            if any(value.ndim != 4 or value.shape[-1] != 3 for value in images):
                raise ValueError("Each source camera must have shape [N,H,W,3]")
            forces = np.asarray(data[forces_key][start:end])
            if forces.shape == (end - start, 15):
                forces = forces.reshape(end - start, 5, 3)
            arrays = {
                "images": np.stack([np.moveaxis(value, -1, 1) for value in images], axis=1),
                "proprio": np.concatenate([np.asarray(data[key][start:end]) for key in proprio_keys], axis=-1),
                "positions": np.asarray(data[positions_key][start:end]),
                "forces": forces,
                "actions": np.asarray(data[actions_key][start:end]),
                "timestamps": np.asarray(data[timestamp_key][start:end], dtype=np.float64),
                "image_timestamps": np.asarray(data[image_timestamps_key][start:end], dtype=np.float64),
            }
            entry = {"id": f"episode_{index:06d}", "file": f"episodes/episode_{index:06d}.npz"}
            np.savez(stage / entry["file"], **arrays)
            _load_episode(stage, entry, metadata)
            entries.append(entry)
            start = end
        manifest = {"schema_version": 1, "metadata": metadata, "episodes": entries,
                    "source_format": "replay_group_with_exclusive_episode_ends"}
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        read_manifest(stage)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output appeared during conversion: {output}")
        os.rename(stage, output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Existing replay_buffer.zarr directory")
    parser.add_argument("--output", type=Path, required=True, help="New canonical dataset directory")
    parser.add_argument("--metadata", type=Path, required=True, help="JSON containing the required metadata object")
    parser.add_argument("--image-keys", nargs="+", default=["oak_rgb", "realsense_rgb"])
    parser.add_argument("--proprio-keys", nargs="+", default=["tcp_state", "q_state"])
    parser.add_argument("--timestamp-key", default="timestamp")
    parser.add_argument("--image-timestamps-key", default="image_timestamps")
    parser.add_argument("--positions-key", default="tactile_pos")
    parser.add_argument("--forces-key", default="tactile")
    parser.add_argument("--actions-key", default="action")
    args = parser.parse_args(argv)
    with args.metadata.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    _validate_metadata(metadata)
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Source and output must be different directories")
    if not args.source.is_dir():
        raise ValueError("Source Zarr directory does not exist")
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("Optional Zarr conversion requires zarr>=2.16,<3; the training core does not require Zarr") from exc
    source = zarr.open_group(str(args.source), mode="r")
    convert_replay_group(source, args.output, metadata, image_keys=args.image_keys,
                         proprio_keys=args.proprio_keys, timestamp_key=args.timestamp_key,
                         image_timestamps_key=args.image_timestamps_key, positions_key=args.positions_key,
                         forces_key=args.forces_key, actions_key=args.actions_key)
    print(f"Wrote validated canonical dataset to {args.output}")


if __name__ == "__main__":
    main()
