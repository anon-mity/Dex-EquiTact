# Recorded episode format

Prepare your recorded demonstrations as a local `manifest.json` and explicitly listed NPZ episode files. Each episode contains synchronized arrays on one fast observation/action grid. The data producer supplies calibrated forces, physical fingertip positions, aligned timestamps, and actions in the declared convention. The loader validates this format and constructs training windows.

## Manifest

```json
{
  "schema_version": 1,
  "metadata": {
    "finger_order": ["thumb", "index", "middle", "ring", "little"],
    "position_frame": "wrist",
    "position_unit": "m",
    "force_frame": "sensor_local",
    "force_unit": "N",
    "force_axes": "common_calibrated",
    "action_convention": "arm_delta_hand_target",
    "action_dim": 26,
    "proprio_dim": 26,
    "num_cameras": 2,
    "timestamp_unit": "s",
    "arm_translation_frame": "base",
    "arm_rotation_representation": "rotation_vector",
    "arm_rotation_composition": "left_multiply",
    "hand_command_unit": "rad",
    "position_source": "wrist_frame_fingertip_tracking",
    "force_calibration_id": "fingertip_force_calibration_v1"
  },
  "episodes": [
    {"id": "episode_000", "file": "episodes/episode_000.npz"},
    {"id": "episode_001", "file": "episodes/episode_001.npz"}
  ]
}
```

The JSON illustrates the schema. Set the arm frame, rotation representation/composition, hand units, position source, and force calibration identifier to the conventions used by your recordings. Those six fields are mandatory nonempty strings. Calibration and coordinate conversion are performed by the data producer before loading; metadata records the resulting convention.

The fixed physical fields above are checked exactly. Input forces must already be calibrated numerical vectors in `sensor_local` coordinates. `common_calibrated` declares consistent local-axis conventions across the five sensors; forces retain their local coordinates. Positions must be measured or computed from validated fingertip kinematics in wrist coordinates, in meters. The `position_source` field identifies that source, and `force_calibration_id` identifies the applied calibration. All five physical positions and forces are required in the declared order. The loader checks array shape, finiteness, and declarations; physical calibration and kinematic validation remain part of data preparation.

Episode IDs and resolved file paths must both be unique. Files must exist inside the dataset directory; absolute paths and paths escaping through `..` or symlinks are rejected. Additional provenance metadata can be stored alongside the required fields.

## Episode arrays

Each `.npz` has these seven required named arrays. Save with `numpy.savez`; loading always uses `allow_pickle=False`.

| Key | Shape | Numerical contract |
| --- | --- | --- |
| `images` | `[N,V,3,H,W]` | RGB uint8 in `[0,255]` or floating values in `[0,1]` |
| `proprio` | `[N,P]` | Finite current robot state; stable documented component order |
| `positions` | `[N,5,3]` | Finite current fingertip positions, wrist frame, meters |
| `forces` | `[N,5,3]` | Finite calibrated 3D force vectors, Newtons |
| `actions` | `[N,A]` | Finite six arm increments plus `A-6` hand targets; `A=26` or `28` |
| `timestamps` | `[N]` | Finite floating seconds, strictly increasing; float64 recommended |
| `image_timestamps` | `[N,V]` | Actual source exposure timestamps in the same clock, floating seconds |

`N` is the episode length, `V=num_cameras`, `P=proprio_dim`, and `A=action_dim`. Every array must match these declared dimensions exactly. Actions are never truncated or padded. The 26D layout has 20 hand targets; the 28D layout has 22. The six arm values contain three translation increments and a declared three-value rotation increment representation. Translation frame, rotation representation and multiplication order, and hand target units must match the actual producer. The loader treats these values numerically and does not implement a robot controller.

If the recorded arm commands are absolute TCP poses, convert them upstream into the required increments using the actual current pose, command timing, pose convention, and frame/composition definitions. Validate that conversion before declaring `arm_delta_hand_target`; the loader does not convert action semantics.

Camera images may be repeated on the fast grid using the latest image already available at each timestamp. Each camera's source timestamps must be nondecreasing and cannot exceed the corresponding fast-grid timestamp. Preserve the repeated image's source timestamp when holding a frame on the grid. The producer must account for exposure-to-availability delay when aligning observations, so each slow context contains only frames available at its anchor. At every fast step, tactile observations must precede the corresponding action decision.

The loader converts uint8 RGB to float32 `[0,1]`, leaves valid floating RGB in that range, and returns all model inputs as CPU float32 tensors. Images are not standardized by the data normalizer. Prepare a consistent camera resolution before saving episodes; the loader applies no resize or augmentation, and the visual encoder requires `H,W >= 8`.

NPZ decoding is episode-granular. Initialization validates one episode at a time. Sampling caches one full decoded episode per DataLoader worker, so memory and decoding cost scale with episode size and worker count. Preserve acquisition episode boundaries and keep related demonstrations together when defining a session- or object-level split.

## Exact temporal indices

Defaults are `observation_horizon=2`, `action_horizon=16`, `prediction_horizon=8`, `slow_stride=16`, `sample_stride=1`.

For anchor index `s`, a sample contains:

| Returned key | Source indices | Default output shape |
| --- | --- | --- |
| `images` | `[s-16,s]` | `[2,V,3,H,W]` |
| `proprio` | `[s-16,s]` | `[2,P]` |
| `positions`, `forces` | `s` through `s+15` | `[16,5,3]` each |
| `actions` | `s` through `s+15` | `[16,A]` |
| `future_forces` | `s+16` through `s+23` | `[8,5,3]` |

The dataset supports positive horizon arguments: the earliest anchor is `(observation_horizon-1)*slow_stride` and the latest is `N-action_horizon-prediction_horizon`. Anchors advance by `sample_stride`. `PolicyConfig` requires horizons `2/16/8`. The slow history ends at `s` and is fixed for that fast block. `future_forces` supplies the force EMA teacher during training; online conditioning uses only the 16 current tactile frames. The sample includes no future images or proprioception.

Only full valid windows are emitted. No boundary frames are repeated, no target masks are needed, and no sample crosses an episode boundary. Default sampling therefore needs at least 40 frames per episode. Short episodes emit no windows; constructing a dataset with no usable window raises an error.

## Splitting and normalization

```python
from dex_equitact.data import EpisodeDataset, Normalizer, read_manifest, split_episode_ids

manifest = read_manifest("/path/to/dataset")
train_ids, val_ids = split_episode_ids("/path/to/dataset", val_fraction=0.2, seed=0)
normalizer = Normalizer.fit("/path/to/dataset", train_ids)
train = EpisodeDataset("/path/to/dataset", train_ids, normalizer=normalizer)
val = EpisodeDataset("/path/to/dataset", val_ids, normalizer=normalizer)
```

The split is deterministic and disjoint by explicit episode ID. A positive validation fraction leaves at least one episode in each set when two or more episodes exist. One episode or `val_fraction=0` returns train-only IDs and an empty validation list; construct a validation dataset only when validation IDs are present. The trainer requires separate training and validation episodes with complete windows. For session- or object-level evaluation, supply corresponding disjoint episode groups through the dataset API.

Statistics are fitted only over all frames in the given training episodes. Held-out episode values cannot influence them. Action and proprioception use per-component population mean and standard deviation; near-constant components use denominator 1. Force and position each use one fixed positive scalar, the reciprocal of the maximum training vector norm (with a small positive floor), and zero spatial offset. Scaling is shared across fingers, time, and xyz. These fixed vector maps commute with rotation; vector norms also keep fitted scales independent of the choice of spatial axes. Position and force scales may differ because their units differ.

The serializable normalizer state contains `schema_version`, `metadata`, `train_episode_ids`, `position_scale`, `force_scale`, `actions_mean`, `actions_std`, `proprio_mean`, and `proprio_std`. Vector scales are multiplicative factors. Save the state with the checkpoint and split; reuse it for validation, resume, and deployment. Passing a normalizer to `EpisodeDataset` requires an exact metadata match, including units, frames, action convention, and calibration/provenance declarations. Equal feature widths alone do not establish compatibility.

```python
restored = Normalizer.from_state_dict(normalizer.state_dict())
p_scaled, f_scaled = restored.normalize_tactile(positions, forces)
q_scaled = restored.normalize_proprio(proprio)
raw_actions = restored.denormalize_actions(predicted_normalized_actions)
normalized = restored.normalize_batch(batch)
```

Normalization methods require floating NumPy arrays or Torch tensors and reject integer nonimage inputs to prevent accidental truncation of means or standard deviations. They preserve NumPy versus Torch array type; Torch parameters follow the input device/dtype. `normalize_batch` returns a new dictionary, scales any present `positions`, `forces`, `future_forces`, `proprio`, and `actions`, and leaves other entries alone. Do not call it again when `EpisodeDataset(..., normalizer=...)` has already normalized the sample.

## Existing Zarr recordings

The optional [scripts/convert_zarr.py](../scripts/convert_zarr.py) entry point reads a replay group with `data/*` and exclusive `meta/episode_ends`, preserves every episode boundary, and writes a canonical dataset directory. It rejects existing output directories and publishes output only after validating every episode. The core training data layer uses NumPy and does not require Zarr.

```bash
PYTHONPATH=src python scripts/convert_zarr.py \
  --source /path/to/replay_buffer.zarr \
  --output /path/to/new_canonical_dataset \
  --metadata /path/to/metadata.json
```

The metadata file contains the `metadata` object itself, without the outer manifest. The CLI requires optional `zarr>=2.16,<3`; `--help` and the training core do not import Zarr. Defaults read `oak_rgb` and `realsense_rgb` as `[N,H,W,3]`, concatenate `tcp_state` and `q_state`, and copy `tactile_pos`, `tactile`, `action`, `timestamp`, and `image_timestamps`. CLI options can explicitly map each of those fields. An already confirmed flat force layout `[N,15]` is reshaped to `[N,5,3]` under the required finger-order declaration.

The converter copies numerical values without calibration or absolute-to-increment action conversion. Inputs must already satisfy the declared units, frames, finger order, timestamps, and `arm_delta_hand_target` convention. Required source fields must exist, every source array must match the final exclusive episode boundary, and every converted episode passes the canonical schema checks before output is written.
