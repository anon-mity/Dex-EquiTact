# Paper-aligned clean implementation

Authoritative manuscript: Dex-EquiTact_95.pdf, SHA256
`c49b2571b4e6110295ceb093aca40c28b61fb5104896645cdb0104b4e52cdaa5`.
The user authorizes implementing its method in an independent clean repository.
The existing working tree remains untouched. No robot execution or remote publication.

## Design and interfaces

Choose a self-contained PyTorch implementation, rather than retaining the old
invariant fingertip encoder or importing the legacy training framework. The old
encoder pools away finger identity and lacks the paper's typed shared field.

Canonical batch: images [B,2,V,3,H,W], proprio [B,2,P], positions and forces
[B,16,5,3], future_forces [B,8,5,3], actions [B,16,A]. Data are normalized by
the dataset normalizer before reaching the policy. P/F use separate positive
scalar scales with zero offset; action/proprio use training-only statistics.
Future forces are exclusively EMA supervision. No padded or cross-episode targets.

Typed vector field [B,T,5,2,Cv,3]. Independent causal position/force encoders.
Scalar FiLM gain [B,5,Cv], shared over time/type/XYZ, applied multiplicatively.
Predictor consumes shared vectors, uses invariant position gates and force vectors
to retain the separate rotation laws. Action expert alone flattens each finger.

Both action self attention and tactile cross attention are causal. All cached slow
tokens are visible. Deterministic DDIM restarts from the cached initial noise prefix
at each tactile arrival; only the newest normalized action is returned.

## Ownership and sequence

1. Model agent: models/vn.py, models/tactile.py and focused geometry tests.
2. Diffusion agent: models/diffusion.py and mask/DDIM tests.
3. Data agent: data.py, data contract docs and dataset/normalizer tests.
4. Root: config, vision, policy, streaming, training/CLI, package/docs/configs,
   integration tests and final review. Shared signatures fixed in task messages.

Each implementer first runs meaningful failing tests, then implementation and
focused tests. CPU interpreter: /home/binghan/miniconda3/envs/vjepa2-312/bin/python.
Root integrates and runs the entire suite plus actual optimizer/checkpoint smoke
tests for both 26D and 28D policies. Independent review follows integration.

## Acceptance

- Independent SO(3) position/force rotations, finger ordering, zero input and causality.
- Real DDIM prefix consistency using fixed context/noise, temporal visibility.
- Future target j+1 through j+8; no target-only input in action conditioning.
- EMA teacher detached, excluded from optimizer, updated after optimizer step.
- Train-only normalizer fitting, strict schema/timestamps/physical metadata, no fake FK.
- Train/evaluate/checkpoint/resume and streaming usable without legacy imports.
- Default hyperparameters and unspecified architectural choices explicitly distinguished
  from paper requirements. No claims of reproduced robot success rates.

## Progress

- [x] Read current PDF and audit old model/data paths.
- [x] Isolate clean repository and agree interfaces from paper.
- [x] Implement modules and focused tests (geometry11, diffusion13, data25).
- [x] Integrate training, checkpoint and streaming workflows.
- [x] Independent review, 60 tests and both embodiment CPU smoke workflows.
- [x] Complete documentation and initialize independent Git repository.
- [x] Prepare reviewed source tree for final clean commit/archive and delivery.
