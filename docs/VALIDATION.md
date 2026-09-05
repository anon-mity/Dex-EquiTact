# Validation record

Verified 2026-09-05 on CPU, Python 3.12.13, PyTorch 2.11.0+cu130, NumPy 2.4.6.
CUDA was unavailable. All verification uses synthetic numerical inputs, not robot
demonstrations. No robot device was accessed. The source PDF hash was rechecked
after implementation and still matches the contract.

## Automated checks

```bash
python -m pytest -q
```

**60 passed**. These cover:

- Independent position/force rotations; five-finger ordering; zero input stability;
  scalar FiLM broadcasting; typed future-head equivariance and gradient flow.
- Encoder and action/predictor causality under future perturbations, actual DDIM
  prefix equivalence, visibility of all five current fingers and slow context.
- Correct forward Gaussian corruption and analytical DDIM terminal update.
- All 128 origin/horizon target indices; teacher causality, no target gradients;
  online encoder updates followed by the actual force-only EMA recurrence.
- Complete 16-step streaming for 26D/28D; original noise and cache unchanged;
  training-only heads replaced by exceptions without breaking inference;
  timestep validation and rejection of a seventeenth action without rollover.
- Strict physical metadata/schema/timestamps, full within-episode windows,
  train-only statistics, no integer normalization truncation, no implicit geometry.
- Conversion core rejection of missing recorded signals and output overwrite.
- Real optimizer/checkpoint/load/resume, with bitwise identical CPU model state
  for continuous training and interrupted/resumed training.

Production-schedule prefix tests use 1000 diffusion levels and 10 DDIM updates,
with tiny channel widths to keep the tests fast. An independent 16-step reproduction
observed maximum absolute difference 1.22e-4 against full-prefix sampling; tests use
atol=2e-4/rtol=2e-4 to allow accumulated float32 roundoff. Prefix equivalence is
numerical, not a claim of bitwise identity across matrix sizes/devices.

## Executable workflows

Both commands completed two actual optimizer/EMA steps, checkpoint loading, and
all 16 streaming actions with finite outputs:

```bash
python -m dex_equitact smoke --output /tmp/dex_smoke_wuji --action-dim 26
python -m dex_equitact smoke --output /tmp/dex_smoke_sharpa --action-dim 28
```

Streamed shapes: `[1,16,26]` and `[1,16,28]`. The smoke network has 47,582 and 47,776
online trainable parameters respectively. These deliberately small models are
software fixtures, not experimental policies.

Both full YAML defaults (`configs/wuji.yaml`, `configs/sharpa.yaml`) separately
completed forward/backward, AdamW, EMA, and reactive inference on synthetic inputs:

| Configuration | Online trainable parameters | Including frozen EMA teacher | Action shape |
|---|---:|---:|---|
| WUJI | 968,874 | 994,130 | `[1,26]` |
| Sharpa | 969,644 | 994,900 | `[1,28]` |

The default model still contains its auxiliary prediction head; these counts are
not stripped deployment counts. The EMA teacher is excluded from the optimizer.

`validate-data` and `evaluate` both ran against the synthetic Sharpa dataset.
The former checked 3 episodes/27 complete windows; the latter processed 9 held-out
windows. Reported losses were finite and are not interpreted as task accuracy.

The package wheel built without downloading dependencies and was installed into
an isolated temporary directory. Its WUJI smoke ran from outside the repository,
so it did not depend on the legacy package or the working tree's `PYTHONPATH`.

## Independent review

One reviewer checked geometry/FiLM separately; another independently checked the
action model, masks, target indexing, EMA and streaming. A separate review covered
data/trainer/CLI/checkpoint integration. No unresolved critical or important issue
was identified after fixes.

The integration review additionally compared 4 continuous steps with 2+resume-to-4,
using different checkpoint/validation frequencies. Every model tensor and AdamW
state tensor matched bitwise on CPU, confirming validation does not perturb the
training random stream. The source-level review does not validate robot execution.

## Remaining empirical validation

- Real demonstration loading and physical checks: no complete policy dataset was
  found at the legacy configuration's data path. Required frames, calibration,
  timestamps and action conventions remain the data producer's responsibility.
- Actual Zarr reading: Zarr was not installed. The conversion core and CLI help
  were tested, but an authentic Zarr file conversion was not performed.
- CUDA execution/determinism, training throughput and robot control latency were
  not measured. The implementation recomputes prefixes and has no real-time guarantee.
- Original hyperparameters, trained weights, paper success rates, data efficiency,
  tactile reliance and collapse mitigation have not been reproduced.

These limits do not change the checked tensor/method contracts. They separate
software correctness from experimental evidence that requires data and hardware.
