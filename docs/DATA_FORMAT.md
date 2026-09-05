# Recorded episode format

The data loader accepts a local `manifest.json` and explicitly listed NPZ episode files. Each episode contains synchronized arrays on one fast observation/action grid. It does not discover arbitrary files, use pickle, call hardware, compute FK, infer missing calibration, convert actions, or invent fingertip positions.

The repository's smoke data are deterministic **synthetic test fixtures**. Passing these checks demonstrates software behavior; it is not evidence of real-data calibration, policy quality, robot control performance, or a particular physical sampling rate.

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
    "position_source": "recorded_stream_and_calibration_identifier",
    "force_calibration_id": "actual_calibration_identifier"
  },
  "episodes": [
    {"id": "episode_000", "file": "episodes/episode_000.npz"},
    {"id": "episode_001", "file": "episodes/episode_001.npz"}
  ]
}
```

The example values for arm frames, rotation representation/composition, hand units, and provenance must be replaced with the actual recording convention. Those six fields are mandatory nonempty strings. They are declarations from the data producer; their presence does **not** prove that calibration or conversion was performed. Synthetic fixtures use `base`, `rotation_vector`, `left_multiply`, `rad`, `synthetic`, `synthetic` respectively and must remain clearly labeled as synthetic.

The fixed physical fields above are checked exactly. Input forces are the already calibrated numerical vectors specified by `sensor_local` / `common_calibrated`; `common_calibrated` declares consistent calibrated axis conventions across the five sensors, not that the loader has transformed them into wrist coordinates. Positions are recorded or independently validated physical fingertip locations in wrist coordinates, in meters. Do not substitute sensor-grid coordinates, a generic hand template, zeros for unavailable geometry, or random values. The code can verify shape and finiteness, but cannot distinguish a fabricated finite tensor from a truthful recording. Numerical coordinate and unit conversion belongs in a separately validated producer with recorded provenance.

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

Legacy absolute TCP commands are not accepted merely because their shape is 26 or 28. An upstream conversion needs the actual current pose, command timing, pose convention, and frame/composition definitions, and must produce truthful `arm_delta_hand_target` data. Renaming the metadata does not perform this conversion.

Camera images may be repeated on the fast grid using the latest image already available at each timestamp. Each camera's source timestamps must be nondecreasing and cannot exceed the corresponding fast-grid timestamp. Record the actual repeated image's source time; do not replace it with the grid time. This check prevents later camera frames from entering a frozen slow context. Delays between exposure and availability require the producer to align against actual availability; exposure timestamps alone cannot prove a frame had arrived in a deployed system.

The loader converts uint8 RGB to float32 `[0,1]`, leaves valid floating RGB in that range, and returns all model inputs as CPU float32 tensors. Images are not standardized by the data normalizer. The model owns its visual preprocessing.

NPZ decoding is episode-granular, not random-access video decoding. Initialization validates one episode at a time. Sampling caches one full decoded episode per DataLoader worker, so memory and decoding cost scale with episode size and worker count. Large recordings should be divided into genuine acquisition episodes without leaking related demonstrations across splits. A future indexed storage adapter can improve image access without changing this contract.

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

With arbitrary positive horizons, the earliest anchor is `(observation_horizon-1)*slow_stride`; the latest is `N-action_horizon-prediction_horizon`. Anchors advance by `sample_stride`. The slow history ends at `s` and is fixed for that fast block. Future force is a training target, not policy observation. The model/trainer must keep `future_forces` out of inference conditioning. No future images or future proprioception are returned.

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

The split is deterministic and disjoint by explicit episode ID. A positive validation fraction leaves at least one episode in each set when two or more episodes exist. One episode or `val_fraction=0` returns train-only IDs and an empty validation list; do not construct a validation dataset from an empty list. Related demonstrations of the same object/session may require a stronger manually supplied split than random episode separation.

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

The training data layer does not depend on Zarr or load the old `ImplicitRDP` package. The optional `scripts/convert_zarr.py` entry point reads an existing replay group with `data/*` and exclusive `meta/episode_ends`, preserves every episode boundary, and writes a new canonical directory. It refuses existing output directories and publishes output only after validating all episodes in a temporary directory.

```bash
PYTHONPATH=src python scripts/convert_zarr.py \
  --source /path/to/replay_buffer.zarr \
  --output /path/to/new_canonical_dataset \
  --metadata /path/to/truthful_metadata.json
```

The metadata file contains the `metadata` object itself, without the outer manifest. The CLI requires optional `zarr>=2.16,<3`; `--help` and the training core do not import Zarr. Defaults read `oak_rgb` and `realsense_rgb` as `[N,H,W,3]`, concatenate `tcp_state` and `q_state`, and copy `tactile_pos`, `tactile`, `action`, `timestamp`, and `image_timestamps`. CLI options can explicitly map each of those fields. An already confirmed flat force layout `[N,15]` is reshaped to `[N,5,3]` under the required finger-order declaration.

This converter copies numerical values and performs no calibration or absolute-to-increment action conversion. It accepts only metadata declaring the already-canonical action convention, with corresponding real input values. Missing `timestamp`, camera source timestamps, positions, or unknown action conventions are acquisition gaps, not fields to fabricate. The legacy repository's converter did not save timestamps and its commands were documented as absolute targets; such files cannot be made valid just by running this command or changing a label. Recover the measured source signals and implement a separately validated semantic conversion first.

The conversion core is tested with small NumPy-backed replay groups, including missing-signal rejection and output protection. The environment used for implementation did not contain Zarr or a real policy replay, so an actual Zarr file conversion and physical data validation remain unverified.
