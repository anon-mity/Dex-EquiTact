# Method implementation map

Source: method equations (1)–(17), Dex-EquiTact_95.pdf (2026-09-05 snapshot).
SHA256: `c49b2571b4e6110295ceb093aca40c28b61fb5104896645cdb0104b4e52cdaa5`.
The PDF defines the method; the adjacent contract records its executable obligations.

| Equations | Implemented requirement | Code |
|---|---|---|
| 1–3 | Two slow observations; fixed context per visual cycle; aligned fast tactile/action slots | `models/vision.py`, `data.py`, `streaming.py` |
| 4–8 | Separate causal position/force VN streams, independent rotation laws | `models/vn.py`, `TypedTactileEncoder` |
| 9 | Ordered `[B,T,5,2,Cv,3]` typed field; no invariant pooled readout | `TypedTactileEncoder.forward` |
| 10 | `[B,5,Cv]` scalar FiLM gain, shared over type/XYZ; direct slow token path | `ScalarFiLM`, `ActionDenoiser` |
| 11 | Causal action self attention and action–touch cross attention; slow tokens always visible | `ActionDenoiser.forward` |
| 12 | Gaussian forward corruption; denoiser predicts epsilon | `DiffusionSchedule.add_noise`, `policy.loss` |
| 13 | Eta=0 DDIM, cached original noise, recompute prefix and return only newest | `DiffusionSchedule.sample`, `ReactiveController` |
| 14 | Shared-state future force prediction, no action input | `FutureForcePredictor` |
| 15–17 | Detached force-only EMA targets; action and latent losses; optimizer then EMA | `policy.py`, `training.py` |

## Paper constants and tensor contract

Two observations, sixteen current tactile/action slots, eight future force targets,
five ordered fingers (thumb, index, middle, ring, little). WUJI actions are 6+20;
Sharpa actions are 6+22. Each embodiment has a separate model.

Position vectors are wrist-frame; forces remain calibrated sensor-local. These are
separate SO(3) types, not a shared physical coordinate system. The action expert is
a scalar transformer; no whole-policy equivariance claim is made. Flattening is
internal to that expert. The future head consumes the same vector field before
flattening. No action-conditioned world model is implemented or claimed.

## Explicit implementation choices

The manuscript does not specify the following details. They are selected defaults,
not the hyperparameters that produced the paper's reported results.

- VN: 32 vector channels, 2 layers, 4 heads; per-token vector RMS norm; channel-only
  linear maps; invariant scalar gates; dot-product attention on complete vectors.
  Learned scalar relative-time and finger-pair biases retain finger identity without
  adding XYZ offsets. Same-time fingers may interact; future times never do.
- Vision: three trainable convolution layers, no pretrained weights, GroupNorm,
  2×2 spatial token grid, camera/time/patch embeddings; proprio MLP tokens. There is
  no fixed input resize or augmentation in the loader: preprocessed RGB resolution
  is part of the recorded dataset. This CNN is replaceable if the original backbone
  specification becomes available.
- FiLM: mean-pool cached context followed by Linear–SiLU–Linear to five scalar
  channel-gain vectors. No vector shift. Negative scalar gain is mathematically
  compatible with equivariance.
- Action expert: width 128, 2 layers, 4 heads; read-only concatenated slow/tactile
  cross-attention memory, causal action self-attention, per-token LayerNorm,
  zero dropout, absolute time/finger embeddings, sinusoidal diffusion-time features.
- Predictor: force channel projection to D×Cv; per-finger position squared-norm
  features feed an MLP and scalar `1+tanh` gate. This choice respects independent
  rotation types. It cannot produce a nonzero equivariant force vector from an
  entirely zero force carrier; it has no action input or recursive rollouts.
- Diffusion: 1000-step linear beta schedule from 1e-4 to 0.02; 10 evenly spaced
  selected DDIM steps, eta=0, no clipping/thresholding. Terminal alpha_bar is about
  4.04e-5, close to the Gaussian prior. Tiny smoke configurations use fewer steps
  only to exercise software; they are not performance settings.
- Loss: sum squared error across action coordinates, then mean batch/time. For
  force, sum finger/channel/XYZ then mean batch/origin/horizon. Lambda=1 by default.
- Teacher: exact initial online force copy, momentum 0.99, updated once after each
  successful optimizer step; no target gradients, always evaluation mode.
- AdamW lr=1e-4, weight decay=1e-6, grad norm clip=1, no mixed precision or learning-rate
  schedule. CLI records settings. CPU tests check exact resume; CUDA reproducibility
  and real-time throughput have not been established.
- Full valid windows only, no boundary padding. All 16 origins predict next 8 force
  states. EMA processes the causal continuation of 24 force frames; final 8 never
  enter the online action/predictor branch. No next-cycle image enters the cache.
- P/F scale by inverse maximum training vector norm, shared over all XYZ/fingers.
  Proprio/actions use training-only per-coordinate mean/std. No geometry centering.
- Slow observation spacing defaults to 16 fast grid samples and is configurable.
  Actual sensor rates and time alignment must come from recorded acquisition.

## Changes from the legacy engineering path

The audited legacy VN path mixed position and force as vector channels, averaged
across fingers and produced a rotation-invariant scalar readout. It also lacked the
typed FiLM/shared future-force training path required here. This clean repository
therefore reconstructs those components instead of wrapping that encoder.

Legacy optional random fingertip geometry and full-replay normalization are removed.
Missing positions/timestamps fail validation. Validation episodes never fit
statistics. Absolute TCP commands are not silently treated as paper arm increments.
Existing trained checkpoints are structurally incompatible and require retraining.

## What this implementation does not establish

EMA self-prediction is not a mathematical guarantee against representation collapse.
Passing geometric/causal/gradient checks does not demonstrate tactile reliance,
data efficiency, task success, robustness, or the paper's attention findings.
Those require the original data, training protocol, and experiments. The repository
does not invent hardware FK, calibration, missing timestamps, trained weights,
success-rate figures, or hard real-time execution guarantees.
