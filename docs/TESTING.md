# Testing

The automated suite contains 60 tests covering geometry, temporal causality, diffusion, data validation, optimization, checkpointing, and streaming. Tests create temporary numerical inputs internally and use small models so they can run on CPU. They validate software behavior; policy performance is evaluated separately on task data and robot rollouts.

From the repository root, install development dependencies and run:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

## Coverage

| Test module | Checks |
| --- | --- |
| [test_geometry.py](../tests/test_geometry.py) | Independent position/force rotations, retained finger ordering, encoder causality, scalar FiLM, future-head transformation laws, gradient flow, and finite zero-input behavior |
| [test_diffusion.py](../tests/test_diffusion.py) | Forward Gaussian corruption, analytical DDIM updates and terminal clean estimate, denoiser masks, visible current fingers and slow context, deterministic sampling, and prefix consistency |
| [test_lifecycle.py](../tests/test_lifecycle.py) | Full 16-step cycles for both action dimensions with the 1,000-level/10-update schedule, unchanged noise/context caches, training-head exclusion, all 128 future-target indices, and causal EMA targets |
| [test_policy.py](../tests/test_policy.py) | Joint losses, online optimizer updates, the force EMA recurrence, prefix streaming, action dimensions, future-target shape, and timestamp boundaries |
| [test_data.py](../tests/test_data.py) | Manifest and physical metadata, array shapes and timestamps, within-episode windows, training-only normalization, metadata compatibility, conversion, and output protection |
| [test_training.py](../tests/test_training.py) | Optimizer/checkpoint/load lifecycle and exact CPU model-state equality between continuous and interrupted/resumed training |

The full-cycle diffusion tests retain the default 1,000 diffusion levels and 10 DDIM updates while reducing network widths. They compare streamed actions with a full-prefix sample using `atol=2e-4` and `rtol=2e-4`, allowing accumulated float32 roundoff across different prefix matrix sizes. Cached context and initial noise are checked for exact equality.

## Focused checks

Run a module independently when changing its behavior:

```bash
python -m pytest -q tests/test_geometry.py
python -m pytest -q tests/test_diffusion.py tests/test_lifecycle.py
python -m pytest -q tests/test_data.py tests/test_training.py
```

The training test performs optimizer and EMA updates, saves and reloads a checkpoint, and resumes from the saved state. The streaming tests exercise all 16 tactile arrivals for both 26D and 28D actions and verify that only the newest action is returned. The Zarr conversion core is tested with temporary NumPy-backed replay groups, including missing-field rejection and output protection.

For user-provided recordings, `validate-data` checks the schema, timestamps, and complete-window availability before training. `evaluate` reports held-out action-noise and future-latent losses using the checkpoint's split and normalizer. See the [README](../README.md) for these commands and [Data format](DATA_FORMAT.md) for the required inputs.
