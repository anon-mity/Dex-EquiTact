# Dex-EquiTact

English | [简体中文](README.zh-CN.md)

An independent PyTorch implementation of Dex-EquiTact for reactive dexterous
manipulation using vision, proprioception and three-axis fingertip forces.
It includes a typed equivariant tactile encoder, causal diffusion policy,
future-force latent supervision, reproducible training and streaming inference.

This release has been verified on synthetic inputs. Training on the original
demonstrations and real-robot success rates have not been reproduced. Datasets,
trained checkpoints and robot drivers are not included.

## Method overview

- **Typed tactile state.** Separate causal vector-neuron streams encode wrist-frame
  fingertip positions and calibrated sensor-local forces under independent SO(3)
  rotations. The state retains both vector types and all five fingers in the order
  **thumb, index, middle, ring, little**: `[B,T,5,2,Cv,3]`.
- **Shared visual conditioning.** Two slow RGB/proprioceptive observations produce
  cached tokens and scalar FiLM gains. FiLM preserves the vector field; the action
  expert also attends directly to the cached slow tokens. Flattening occurs inside
  the action expert, so equivariance is a property of the typed representation,
  not a claim about the complete action policy.
- **Causal reactive diffusion.** The denoiser predicts injected noise (epsilon),
  with causal action–action and action–tactile attention. Deterministic DDIM
  (`eta=0`) resamples the arrived prefix using cached original noise and context.
  At each tactile arrival, only the newest action is returned for execution.
- **Future-force learning.** Each of 16 tactile origins predicts the next 8 force
  latent states from the shared FiLM representation. A force-only EMA encoder
  provides detached targets and updates after each optimizer step. The predictor
  and teacher are used only during training; neither requires action input.

```mermaid
flowchart LR
    S[Two RGB / proprio observations] --> C[Cached slow tokens]
    P[Wrist-frame positions] --> T[Separate causal VN streams]
    F[Sensor-local forces] --> T
    T --> V[Typed five-finger state]
    C --> M[Scalar FiLM]
    V --> M
    C --> A[Causal epsilon denoiser]
    M --> A
    A --> D[DDIM with cached noise]
    D --> N[Newest action]
    A -. training only .-> AL[Action noise loss]
    X[Injected noise] -. target .-> AL
    M -. training only .-> H[Future-force predictor]
    Q[Recorded current + future forces] -. training only .-> E[Force EMA teacher]
    H -.-> L[Latent loss]
    E -. detached targets .-> L
```

The CNN, network widths, optimizer and diffusion schedule use explicit engineering
defaults where the source manuscript leaves details unspecified. See the
[method map](docs/METHOD_ALIGNMENT.md) and
[implementation contract](docs/IMPLEMENTATION_CONTRACT.md) for the exact choices
and manuscript provenance.

## Installation and quick start

Requires Python **3.10+**, PyTorch **2.2+**, NumPy and PyYAML. The package uses a
trainable CNN and does not download pretrained weights.

```bash
git clone https://github.com/anon-mity/Dex-EquiTact.git
cd Dex-EquiTact
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

python -m dex_equitact smoke --output /tmp/dex_smoke_wuji --action-dim 26
python -m dex_equitact smoke --output /tmp/dex_smoke_sharpa --action-dim 28
```

Each smoke command creates labeled synthetic data, performs two optimizer/EMA
updates, saves and reloads a checkpoint, then runs all 16 streaming steps with a
small test model. Expected action shapes are `[1,16,26]` and `[1,16,28]`;
`smoke_result.json` records the result. Use a new or empty output directory.

The installed CLI is also available as `dex-equitact`. Without an editable install,
run from the repository root with `PYTHONPATH=src python -m dex_equitact ...`.
Use `python -m dex_equitact train --help` to inspect training options.

## Embodiment profiles

Train a separate policy for each embodiment:

| Profile | Configuration | Action layout | Proprioception width |
| --- | --- | --- | --- |
| WUJI | [wuji.yaml](configs/wuji.yaml) | 6 arm increments + 20 hand targets = 26 | 26 |
| Sharpa | [sharpa.yaml](configs/sharpa.yaml) | 6 arm increments + 22 hand targets = 28 | 28 |

Both profiles use two cameras, two slow observations, 16 tactile/action slots and
8 future-force target steps. Default networks use 32 vector channels and an action
transformer width of 128, with 1000 training diffusion levels and 10 DDIM updates.

## Prepare recorded data

Follow the [recorded data schema](docs/DATA_FORMAT.md): a `manifest.json` explicitly
lists NPZ episodes containing `images`, `proprio`, `positions`, `forces`, `actions`,
`timestamps` and `image_timestamps`. Metadata declares finger order, coordinate
frames, units, calibration provenance and the complete action convention.

Positions must be recorded or independently validated wrist-frame fingertip
locations in meters; forces are calibrated sensor-local vectors in Newtons.
Geometry, missing timestamps and absolute-to-increment action conversion must be
supplied by the data producer. The [optional Zarr adapter](scripts/convert_zarr.py)
is described in the schema document.

Training splits whole episodes and fits normalization only on training episodes.
Position and force each use one scalar shared across XYZ, fingers and time, with
no vector offset; actions and proprioception use per-component mean/std.
Training needs separate usable train/validation episodes; the default window
requires at least 40 aligned frames. `--slow-stride` defaults to 16 fast-grid
samples between slow observations. Set it from the actual acquisition protocol;
it does not declare a physical sampling frequency.

## Train, resume and evaluate

Replace `/path/to/dataset` with a dataset prepared above. Use `--device cpu` for
CPU execution, or `--device cuda` with a working CUDA installation.

```bash
python -m dex_equitact validate-data --data /path/to/dataset

python -m dex_equitact train --config configs/wuji.yaml \
  --data /path/to/dataset --output runs/wuji --device cuda \
  --steps 100000 --batch-size 8 --save-every 1000

python -m dex_equitact train --config configs/wuji.yaml \
  --data /path/to/dataset --output runs/wuji --device cuda \
  --steps 200000 --batch-size 8 --save-every 1000 --resume runs/wuji/last.pt

python -m dex_equitact evaluate --data /path/to/dataset \
  --checkpoint runs/wuji/last.pt --device cuda
```

These step counts are usage examples, not recovered experimental settings.
`--steps` is the **target total optimizer step count**, including resumed steps.
Keep the model configuration, data and recorded training settings unchanged when
resuming. For Sharpa, use `configs/sharpa.yaml` and a separate output directory.

`last.pt` stores online/EMA weights, optimizer and random states, normalization,
episode split, configuration and a dataset content fingerprint. `run.json` records
provenance and `metrics.jsonl` logs losses, including validation at each checkpoint.
Evaluation uses the checkpoint's held-out split and reports action-denoising and
latent losses. Task success and robot execution are separate evaluations.

## Streaming API

The example below assumes observations have already been acquired. Use float32
tensors on the policy's device, with `B` batch items, `V` cameras and `P`
proprioceptive components matching the checkpoint; `H,W` must each be at least 8.

```python
from dex_equitact.training import load_policy
from dex_equitact.streaming import ReactiveController

policy, normalizer, checkpoint = load_policy("runs/wuji/last.pt", device="cpu")
controller = ReactiveController(policy, normalizer)  # policy is already in eval mode

# Two already available slow observations:
# images: [B,2,V,3,H,W], RGB in [0,1]; proprio: [B,2,P], recorded physical units.
controller.start_cycle(images, proprio, timestamp=visual_time)

# Call once for each newly arrived tactile frame, in the declared finger order.
# positions: [B,5,3], wrist-frame meters; forces: [B,5,3], sensor-local Newtons.
latest_action = controller.step(positions, forces, timestamp=tactile_time)
# [B,26] or [B,28], denormalized to the recorded action convention.
# The caller executes this newest action.
```

With the checkpoint normalizer, supply physical-unit proprioception/tactile input
and receive physical-unit actions. With `ReactiveController(policy)` instead, all
nonimage inputs and outputs are **normalized values**. RGB always remains `[0,1]`.

Pass finite timestamps in seconds. Each tactile timestamp must be at or after its
cycle start and strictly newer than the preceding tactile timestamp. A cycle
accepts at most 16 `step` calls; call `start_cycle` when a new slow observation
arrives to refresh context, clear tactile history and cache new noise.
The caller owns hardware I/O and interprets arm translation frames, rotation
composition and hand target units exactly as declared in the dataset manifest.

## Verification

```bash
python -m pytest -q
```

The [validation record](docs/VALIDATION.md) reports **60 passing CPU tests** on
Python 3.12 / PyTorch 2.11, covering typed rotations, finger ordering, causality,
DDIM prefix consistency, EMA targets, both streaming profiles, data contracts and
checkpoint/resume behavior. These are software checks with synthetic inputs;
CUDA execution and real-robot latency/performance remain unverified.

## Repository map

| Area | Source |
| --- | --- |
| Typed VN encoder, FiLM, future-force head | [vn.py](src/dex_equitact/models/vn.py), [tactile.py](src/dex_equitact/models/tactile.py) |
| Slow visual/proprioceptive tokens | [vision.py](src/dex_equitact/models/vision.py) |
| Causal denoiser and DDIM | [diffusion.py](src/dex_equitact/models/diffusion.py) |
| Shared state, joint loss and EMA | [policy.py](src/dex_equitact/policy.py) |
| Reactive inference and cycle cache | [streaming.py](src/dex_equitact/streaming.py) |
| Data, normalization and training lifecycle | [data.py](src/dex_equitact/data.py), [training.py](src/dex_equitact/training.py) |
| Configuration, CLI and tests | [config.py](src/dex_equitact/config.py), [cli.py](src/dex_equitact/cli.py), [tests/](tests) |

## License

Released under the [MIT License](LICENSE).
