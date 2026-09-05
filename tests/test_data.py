"""Deterministic test fixtures exercise the contract, not real robot data."""
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def api():
    try:
        return importlib.import_module("dex_equitact.data")
    except ModuleNotFoundError as exc:
        detail = str(exc)
        class MissingPipeline:
            def __getattr__(self, name):
                pytest.fail(f"Strict data pipeline is not implemented: {detail}")
        return MissingPipeline()


def make_root(tmp_path, action_dim=26, n_frames=42, n_episodes=2):
    metadata = {
        "finger_order": ["thumb", "index", "middle", "ring", "little"],
        "position_frame": "wrist", "position_unit": "m",
        "force_frame": "sensor_local", "force_unit": "N",
        "force_axes": "common_calibrated",
        "action_convention": "arm_delta_hand_target",
        "action_dim": action_dim, "proprio_dim": 7, "num_cameras": 2,
        "timestamp_unit": "s",
        "arm_translation_frame": "base",
        "arm_rotation_representation": "rotation_vector",
        "arm_rotation_composition": "left_multiply",
        "hand_command_unit": "rad",
        "position_source": "synthetic",
        "force_calibration_id": "synthetic",
    }
    episodes = []
    for episode in range(n_episodes):
        t = np.arange(n_frames, dtype=np.float32)
        arrays = {
            "images": np.broadcast_to(t[:, None, None, None, None].astype(np.uint8),
                                      (n_frames, 2, 3, 4, 4)).copy(),
            "proprio": np.broadcast_to(t[:, None] + episode * 1000, (n_frames, 7)).copy(),
            "positions": np.broadcast_to((t / 100)[:, None, None], (n_frames, 5, 3)).copy(),
            "forces": np.broadcast_to((t + episode * 1000)[:, None, None], (n_frames, 5, 3)).copy(),
            "actions": np.broadcast_to((t + episode * 1000)[:, None], (n_frames, action_dim)).copy(),
            "timestamps": np.arange(n_frames, dtype=np.float64) / 100,
            "image_timestamps": np.broadcast_to(np.arange(n_frames, dtype=np.float64)[:, None] / 100,
                                                (n_frames, 2)).copy(),
        }
        name = f"episode_{episode}.npz"
        np.savez(tmp_path / name, **arrays)
        episodes.append({"id": f"episode_{episode}", "file": name})
    manifest = {"schema_version": 1, "metadata": metadata, "episodes": episodes}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def modify_episode(root, **changes):
    path = root / "episode_0.npz"
    with np.load(path, allow_pickle=False) as source:
        arrays = dict(source)
    arrays.update(changes)
    np.savez(path, **arrays)


@pytest.mark.parametrize("action_dim", [26, 28])
def test_full_windows_preserve_dimensions_and_future_boundary(api, tmp_path, action_dim):
    make_root(tmp_path, action_dim=action_dim)
    dataset = api.EpisodeDataset(tmp_path, ["episode_0"])
    assert len(dataset) == 3  # starts 16,17,18, each ending at start+24
    item = dataset[0]
    assert item["images"].shape == (2, 2, 3, 4, 4)
    assert item["proprio"].shape == (2, 7)
    assert item["positions"].shape == item["forces"].shape == (16, 5, 3)
    assert item["actions"].shape == (16, action_dim)
    assert item["future_forces"].shape == (8, 5, 3)
    torch.testing.assert_close(item["proprio"][:, 0], torch.tensor([0., 16.]))
    torch.testing.assert_close(item["actions"][:, 0], torch.arange(16., 32.))
    torch.testing.assert_close(item["future_forces"][:, 0, 0], torch.arange(32., 40.))
    assert dataset[-1]["future_forces"][-1, 0, 0] == 41


def test_no_windows_cross_episode_boundaries(api, tmp_path):
    make_root(tmp_path)
    data = api.EpisodeDataset(tmp_path, ["episode_0", "episode_1"], sample_stride=2)
    assert len(data) == 4
    assert data[1]["actions"][0, 0] == 18
    assert data[2]["actions"][0, 0] == 1016


@pytest.mark.parametrize("key,value", [
    ("position_frame", "unknown"),
    ("force_axes", "uncalibrated"),
    ("finger_order", ["index", "thumb", "middle", "ring", "little"]),
    ("action_convention", "absolute_tcp_target"),
])
def test_physical_semantics_must_be_explicit(api, tmp_path, key, value):
    manifest = make_root(tmp_path)
    manifest["metadata"][key] = value
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=key):
        api.read_manifest(tmp_path)


def test_missing_geometry_rejected_without_placeholder(api, tmp_path):
    make_root(tmp_path)
    with np.load(tmp_path / "episode_0.npz") as source:
        arrays = {k: source[k] for k in source.files if k != "positions"}
    np.savez(tmp_path / "episode_0.npz", **arrays)
    with pytest.raises(ValueError, match="positions"):
        api.EpisodeDataset(tmp_path, ["episode_0"])


@pytest.mark.parametrize("failure", ["nan", "shape", "time", "future_image", "action_width"])
def test_corrupt_episode_is_rejected(api, tmp_path, failure):
    make_root(tmp_path)
    if failure == "nan":
        values = np.zeros((42, 5, 3), np.float32)
        values[10, 2, 0] = np.nan
        modify_episode(tmp_path, positions=values)
    elif failure == "shape":
        modify_episode(tmp_path, positions=np.zeros((42, 15), np.float32))
    elif failure == "time":
        modify_episode(tmp_path, timestamps=np.zeros(42, np.float64))
    elif failure == "future_image":
        values = np.broadcast_to(np.arange(42)[:, None] / 100, (42, 2)).copy()
        values[16, 0] += 0.001
        modify_episode(tmp_path, image_timestamps=values)
    else:
        modify_episode(tmp_path, actions=np.zeros((42, 28), np.float32))
    with pytest.raises(ValueError):
        api.EpisodeDataset(tmp_path, ["episode_0"])


def test_held_camera_frames_and_float_images_are_supported(api, tmp_path):
    make_root(tmp_path)
    held_timestamps = np.broadcast_to((np.arange(42) // 16 * 16)[:, None] / 100, (42, 2)).copy()
    images = np.full((42, 2, 3, 4, 4), 0.4, dtype=np.float32)
    modify_episode(tmp_path, image_timestamps=held_timestamps, images=images)
    item = api.EpisodeDataset(tmp_path, ["episode_0"])[0]
    torch.testing.assert_close(item["images"], torch.full_like(item["images"], 0.4))


def test_split_deterministic_disjoint_and_nonempty(api, tmp_path):
    make_root(tmp_path, n_episodes=5)
    first = api.split_episode_ids(tmp_path, val_fraction=0.4, seed=3)
    assert first == api.split_episode_ids(tmp_path, val_fraction=0.4, seed=3)
    train, val = first
    assert len(train) == 3 and len(val) == 2
    assert set(train).isdisjoint(val)
    assert set(train + val) == {f"episode_{i}" for i in range(5)}


def test_train_only_normalization_roundtrip_and_rotation(api, tmp_path):
    make_root(tmp_path)
    normalizer = api.Normalizer.fit(tmp_path, ["episode_0"])
    state = normalizer.state_dict()
    assert state["train_episode_ids"] == ["episode_0"]
    restored = api.Normalizer.from_state_dict(json.loads(json.dumps(state)))
    train = api.EpisodeDataset(tmp_path, ["episode_0"])[0]
    val = api.EpisodeDataset(tmp_path, ["episode_1"])[0]
    norm_val = restored.normalize_batch(val)
    assert torch.min(norm_val["actions"]) > 10
    assert torch.min(norm_val["forces"]) > 10
    torch.testing.assert_close(restored.denormalize_actions(norm_val["actions"]), val["actions"])
    torch.testing.assert_close(val["actions"][0], torch.full((26,), 1016.))
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    p, f = restored.normalize_tactile(train["positions"], train["forces"])
    rp, rf = restored.normalize_tactile(train["positions"] @ rotation.T, train["forces"] @ rotation.T)
    torch.testing.assert_close(rp, p @ rotation.T)
    torch.testing.assert_close(rf, f @ rotation.T)
    normalized_data = api.EpisodeDataset(tmp_path, ["episode_1"], normalizer=restored)[0]
    torch.testing.assert_close(normalized_data["actions"], norm_val["actions"])
    np.testing.assert_allclose(restored.denormalize_actions(norm_val["actions"].numpy()), val["actions"].numpy(), rtol=1e-6)


def test_normalizer_rejects_empty_or_unknown_training_ids(api, tmp_path):
    make_root(tmp_path)
    for ids in [[], ["missing"]]:
        with pytest.raises(ValueError):
            api.Normalizer.fit(tmp_path, ids)


def test_missing_metadata_duplicate_ids_and_path_escape_rejected(api, tmp_path):
    manifest = make_root(tmp_path)
    del manifest["metadata"]["position_frame"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="position_frame"):
        api.read_manifest(tmp_path)
    manifest = make_root(tmp_path)
    manifest["episodes"][1]["id"] = "episode_0"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate"):
        api.read_manifest(tmp_path)
    manifest = make_root(tmp_path)
    manifest["episodes"][0]["file"] = "../outside.npz"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inside"):
        api.read_manifest(tmp_path)


def test_normalizer_metadata_mismatch_rejected(api, tmp_path):
    manifest = make_root(tmp_path)
    normalizer = api.Normalizer.fit(tmp_path, ["episode_0"])
    manifest["metadata"]["force_calibration_id"] = "different_calibration"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="metadata"):
        api.EpisodeDataset(tmp_path, ["episode_1"], normalizer=normalizer)


def test_normalizer_rejects_integer_inputs_instead_of_truncating_statistics(api, tmp_path):
    make_root(tmp_path)
    normalizer = api.Normalizer.fit(tmp_path, ["episode_0"])
    for function, values in [
        (normalizer.normalize_proprio, torch.zeros(2, 7, dtype=torch.int64)),
        (normalizer.denormalize_actions, np.zeros((2, 26), dtype=np.int64)),
        (lambda value: normalizer.normalize_batch({"actions": value}), torch.zeros(2, 26, dtype=torch.int64)),
        (lambda value: normalizer.normalize_tactile(value, value), np.zeros((2, 5, 3), dtype=np.int64)),
        (lambda value: normalizer.normalize_batch({"future_forces": value}), torch.zeros(2, 5, 3, dtype=torch.int64)),
    ]:
        with pytest.raises(ValueError, match="floating"):
            function(values)


def load_converter():
    path = Path(__file__).parents[1] / "scripts" / "convert_zarr.py"
    if not path.exists():
        pytest.fail("Strict Zarr conversion entry point is not implemented")
    spec = importlib.util.spec_from_file_location("convert_zarr", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay_fixture(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    manifest = make_root(source_dir, n_episodes=1)
    with np.load(source_dir / "episode_0.npz") as values:
        source = {
            "data": {
                "oak_rgb": np.moveaxis(values["images"][:, 0], 1, -1),
                "realsense_rgb": np.moveaxis(values["images"][:, 1], 1, -1),
                "tcp_state": values["proprio"][:, :6],
                "q_state": values["proprio"][:, 6:],
                "tactile_pos": values["positions"],
                "tactile": values["forces"].reshape(42, 15),
                "action": values["actions"],
                "timestamp": values["timestamps"],
                "image_timestamps": values["image_timestamps"],
            },
            "meta": {"episode_ends": np.array([42], dtype=np.int64)},
        }
    return source, manifest["metadata"]


def test_strict_converter_preserves_arrays_and_avoids_overwrite(api, tmp_path):
    converter = load_converter()
    source, metadata = replay_fixture(tmp_path)
    destination = tmp_path / "converted"
    converter.convert_replay_group(source, destination, metadata)
    manifest = api.read_manifest(destination)
    sample = api.EpisodeDataset(destination, [manifest["episodes"][0]["id"]])[0]
    torch.testing.assert_close(sample["forces"][:, 0, 0], torch.arange(16., 32.))
    torch.testing.assert_close(sample["images"][1], torch.full_like(sample["images"][1], 16 / 255))
    with pytest.raises(FileExistsError):
        converter.convert_replay_group(source, destination, metadata)


@pytest.mark.parametrize("missing", ["timestamp", "image_timestamps", "tactile_pos"])
def test_converter_does_not_invent_missing_signals(api, tmp_path, missing):
    converter = load_converter()
    source, metadata = replay_fixture(tmp_path)
    del source["data"][missing]
    destination = tmp_path / "converted"
    with pytest.raises(ValueError, match=missing):
        converter.convert_replay_group(source, destination, metadata)
    assert not destination.exists()


def test_converter_rejects_absolute_action_semantics(api, tmp_path):
    converter = load_converter()
    source, metadata = replay_fixture(tmp_path)
    metadata["action_convention"] = "absolute_tcp_target"
    destination = tmp_path / "converted"
    with pytest.raises(ValueError, match="action_convention"):
        converter.convert_replay_group(source, destination, metadata)
    assert not destination.exists()
