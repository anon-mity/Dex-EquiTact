# Dex-EquiTact implementation contract, PDF version 95

Source of truth: `Dex-EquiTact_95.pdf`, Method, printed PDF pages 3–4, equations (1)–(17). SHA256 checked during this review: `c49b2571b4e6110295ceb093aca40c28b61fb5104896645cdb0104b4e52cdaa5`.

This contract does not change the manuscript or propose a new algorithm. It separates **paper requirements**, **necessary consequences of those requirements**, and **implementation choices omitted by the paper**. This implementation resolves the paper's overloaded token notation: `V_tilde` always denotes the FiLM-modulated vector field. The scalar-token flatten/projection is internal to the action expert. The predictor receives the same vector field, has no action condition, and is training-only.

## 1. Tensor and coordinate contract

Use the following canonical axes; alternative internal layouts must preserve the same meanings:

| Symbol / tensor | Batched shape | Meaning |
|---|---|---|
| `P` | `[B, T, 5, 3]` | Ordered fingertip positions expressed in the wrist frame. |
| `F` | `[B, T, 5, 3]` | Ordered calibrated resultant forces, each in its own fingertip sensor's local frame. All sensors use the calibrated common local-axis convention. |
| `V_p`, `V_f` | `[B, T, 5, Cv, 3]` | Position-type and force-type causal VN fields. |
| `V`, `V_tilde` | `[B, T, 5, 2, Cv, 3]` | Stacked typed fields; type index 0 is position, type index 1 is force. |
| `Gamma` | `[B, 5, Cv]` | Scalar FiLM gains from pooled slow context, shared across the two vector types and XYZ. |
| `slow_tokens` | `[B, Nslow, Cslow]` | Direct cached slow conditioning for the action denoiser. |
| internal action tactile tokens | `[B, T, 5, Caction]` | Flatten `[2, Cv, 3]` independently for each finger and project internally. |
| `A`, `epsilon`, denoiser output | `[B, Ha, da]` | Demonstrated actions, injected Gaussian noise, predicted Gaussian noise. |
| `force_prediction` | `[B, Torigin, D, 5, Cv, 3]` | The next `D` force-type latents predicted at each causal origin. |
| `force_target` | same as prediction | Stop-gradient EMA force encoder targets. |

Fixed paper values: `Ho = 2`, `Ha = 16`, `D = 8`, five fingers in thumb/index/middle/ring/little order. `da = 26` for WUJI (6 arm + 20 hand) and `da = 28` for Sharpa Wave (6 arm + 22 hand). Train distinct policies for the embodiments.

Coordinate requirements:

1. Do not rotate local forces into the wrist frame and claim that this reproduces equation (5). Doing so would implement a different coordinate contract. Do not substitute an aggregate wrist force/torque for the five local forces.
2. Position and force have separate normalizations, each using a scalar shared across XYZ. The simplest faithful normalization is `P / position_scale` and `F / force_scale`. Do not apply axis-wise standardization or subtract a fixed nonzero three-vector inside the equivariance claim. Sensor bias removal/calibration before the model is a separate documented physical preprocessing step.
3. `Rp` rotates every wrist-frame position row; `Rf` independently rotates every local force row under the common convention: `(P @ Rp.T, F @ Rf.T)`. This is `SO(3)p x SO(3)f`, not a shared physical rotation of every policy input and not five independently chosen force rotations.
4. Preserve the finger axis and output five ordered carriers. Do not reduce fingers to their mean/sum, max, or one global invariant readout. Internal attention may mix information while retaining an output at each ordered finger, if that architecture is explicitly declared.
5. Scalar position/time/finger labels may influence equivariant scalar weights or attention logits. A learned nonzero XYZ positional/finger bias must not be added to VN vectors.

## 2. Equation-by-equation obligations

| Equation | Executable obligation |
|---|---|
| (1) | Compute slow context once from the most recent `Ho` multi-view image/proprioception observations available at visual cycle start. Cache both slow tokens and the pooled context needed by FiLM until the next visual update. A future image or future proprioception sample cannot enter this cache. |
| (2) | The representation at fast step `j` reads only tactile frames `1:j`. Batch processing of a full training sequence is allowed only if it is functionally equal to processing every available prefix separately. |
| (3) | Pair tactile block `j` with action slot `j`; the block is observed before the corresponding action decision. Train over `Ha` aligned actions; online execution returns only the newest slot. Validate embodiment action dimensions. |
| (4) | All operations inside the claimed vector encoder satisfy the rotation transformation law. Use equivariant channel mixing, nonlinearities, normalization, and attention. Arbitrary XYZ affine layers, coordinate-wise activations, vector biases, and axis-wise normalization are invalid inside this claim. |
| (5) | Input exactly five ordered wrist-frame positions and five local-frame 3D resultant forces sampled at the same fast instant. The data adapter documents units, calibration, ordering, timestamp alignment, and origin of fingertip positions. Placeholder geometric templates are not physical FK. |
| (6) | Treat position and force as independently transforming types. A position vector cannot be added to or linearly mixed with a force vector before the scalar action interface. |
| (7) | Use two causal VN streams with outputs `[5, Cv, 3]` per time step. Each stream must be causal before fusion; a later mask in the action decoder cannot undo future leakage already present in tactile embeddings. |
| (8) | Rotating `P` by `Rp` rotates only `V_p`; rotating `F` by `Rf` rotates only `V_f`. Each branch must remain independent of the other branch's rotation. |
| (9) | Stack the two fields into `[5, 2, Cv, 3]`; do not concatenate their XYZ coordinates into a scalar embedding at this point. Keep all five fingers. |
| (10) | Compute `Gamma = g(pooled_C)` as `[5, Cv]` type-0 scalars. Apply `V_tilde = (1 + Gamma) * V` with broadcasting over time, type, and XYZ. No additive XYZ shift. Keep `V_tilde` as vectors for both experts; flatten/project only inside the action expert. Keep slow tokens as direct action-denoiser conditioning as well as using their pooled context for FiLM. |
| (11) | Both action–action and action–tactile attention allow time `u <= r` and forbid `u > r`. All five fingers at tactile time `u` share visibility. Slow context remains available to every action query. Any conditioning encoder or feedback path must obey the same no-future-dependence property. |
| (12) | Sample Gaussian `epsilon` with the action shape and a diffusion time `k`; construct `A_k = sqrt(alpha_bar[k]) * A + sqrt(1-alpha_bar[k]) * epsilon`. Train `D_theta` to predict `epsilon`, not clean action `A`. Use the same denoiser during DDIM sampling. Full action/tactile sequences may be provided in training only under complete causal masking. |
| (13) | At visual-cycle start cache initial Gaussian noise for all `Ha` action slots. On tactile arrival `j`, deterministically sample the prefix from the original cached noise prefix, cached slow context, and arrived `V_tilde[1:j]`. Recompute earlier slots but execute only `sampled_prefix[-1]` exactly once. Do not execute an entire predicted chunk. |
| (14) | Predict the next `D=8` per-finger force latents from the shared causal FiLM-modulated state at origin `j`. `P_psi` must not consume demonstrated/noisy/predicted actions, future tactile frames, future visual frames, or future target embeddings. It receives the vector field before the action expert's flatten/projection. |
| (15) | Compare predictions with future outputs of an EMA copy of the **force encoder**. Targets are force-type vectors with five fingers retained, not a position-force stack, aggregate force, raw forces, scalar projected action tokens, or a FiLM-modulated online target. Teacher targets are stop-gradient. |
| (16) | Compute denoising squared error against the sampled Gaussian noise, with valid action masks if padding exists. Do not silently replace epsilon prediction with `x0` or velocity prediction. |
| (17) | Optimize `L_act + lambda * L_pred` through all participating online modules. Teacher parameters are excluded from the optimizer. The force prediction head and EMA teacher are training-only and their output never feeds online control. |

The paper's last sentence before equation (11) reuses `V_tilde` for scalar action tokens. The public code interface resolves this by retaining the vector state and performing scalar tokenization inside the action expert.

## 3. Precise temporal mask contract

The paper uses 1-based fast indices `r,u = 1,...,Ha`. For 0-based arrays the same condition is `u <= r`.

- Action self-attention allowed mask: `[Taction, Taction]`, `allow_aa[r,u] = (u <= r)`.
- Action to tactile allowed mask: `[Taction, Ttactile * 5]`, `allow_at[r, 5*u+i] = (u <= r)` for `i=0,...,4`.
- Tactile temporal attention, when flattening time and finger axes: `allow_tt[5*r+i, 5*u+l] = (u <= r)`. All fingers at the same instant are mutually visible; arbitrary finger order should not create an accidental within-frame causal order.
- Slow-condition columns are always allowed, since the cache contains only observations already present at cycle start.
- Padding validity is combined with temporal visibility. Padded targets and padded actions must not contribute to their losses. The handling of missing finger measurements is an explicit data-policy choice; zero-padding an invalid finger without a validity contract is not a calibrated no-contact measurement.
- Attention APIs disagree on whether boolean `True` means allowed or blocked. The implementation must convert from this semantic allowed mask at each API boundary and test observed behavior, not infer correctness from a triangular-looking tensor.

Masks must protect every temporal path. Examples of hidden violations: a bidirectional tactile or condition encoder; future action tokens updating shared slow/tactile tokens that are read back by past actions; pooling/normalization across sequence length; a positional encoding rescaled by current prefix length. Standard per-token LayerNorm is acceptable in the scalar action expert. Vector-layer normalization must remain equivariant and may not mix information from future times.

## 4. Training sample, target visibility, and indexing

One training sample is anchored to a visual update. Slow observations end no later than that anchor. For action slot `j`, the aligned tactile block `T_j` must be available before action `a_j` is issued. The paper does not specify the exact timestamp rule; the dataset contract must make it explicit, including how actions are sampled or held between clocks.

For each selected origin `j` and horizon `d in 1,...,D`, the canonical target is

`target[j,d] = stop_grad(E_force_EMA(F[1:j+d])[j+d])`.

This target uses the future force input through its own target time `j+d`, which is legitimate supervision. It must not read force after `j+d`, and it must never be passed into the online predictor or action path. The online predictor for origin `j` sees `V_tilde[j]`, which can summarize only `T[1:j]`. The target encoder is the force stream only; it has no position, action, FiLM, or slow-context condition.

To predict at **all 16 action origins** with `D=8`, the training loader needs up to **24 tactile/force samples** from the same episode. The final eight are target-only; computing online action conditioning from all 24 and relying on downstream masking is unnecessarily leak-prone. A clean interface receives current tactile inputs separately from target-only future forces.

The paper does not state how its `j+d > Ha` notation crosses visual-cycle boundaries, nor whether predictions are supervised at every action origin. These must be recorded as implementation choices. A faithful practical choice is to treat the extra eight force frames as a causal continuation of the anchored training window for the target stream, keeping them out of the online 16-step condition. An alternative is to supervise only origins with available within-window future frames, using an explicit valid-origin/horizon mask. Never silently wrap indices, repeat terminal frames as valid future targets, cross episode boundaries, or insert next-cycle vision into current-cycle conditioning.

An EMA target for future time `t` can be computed in one causally masked pass over the complete force target window, provided prefix equivalence is tested. Running a bidirectional teacher would change the semantic target from a causal force latent into a future-context-dependent representation, contrary to its being an EMA copy of the causal force stream.

Loss reduction is not numerically specified. A reproducible default matching the written squared Frobenius form is: sum squared feature error over `[finger, channel, XYZ]` per `(batch, origin, horizon)`, then average valid items; similarly sum action dimensions and average valid action positions. Other reductions require explicit documentation and a correspondingly chosen `lambda`; do not switch sum/mean conventions invisibly.

## 5. EMA and gradient contract

Required behavior:

1. Initialize the EMA force encoder as an exact copy of the online force encoder, with matching architecture and state. No random independent teacher initialization.
2. Set teacher parameters `requires_grad=False`; compute teacher outputs under no-grad and detached from the loss graph. Keep the target encoder deterministic (evaluation mode) unless a documented design explicitly needs stochastic targets.
3. Optimize online modules first. On every successful optimizer update, update teacher parameters under no-grad with `theta_teacher <- m * theta_teacher + (1-m) * theta_force_online`.
4. The paper does not specify `m`, a schedule, or update cadence. Expose and record these as implementation choices. With gradient accumulation, updates should be tied to optimizer steps, not each microbatch. A skipped optimizer step should not be counted as a successful update.
5. Handle nonparameter buffers explicitly. Prefer branch normalization without running statistics for causal prefix compatibility. If running buffers exist, declare whether they are copied or EMA-updated and checkpoint them. Never blindly apply floating-point EMA to integer counters.
6. Save and restore both online force and EMA force state, optimizer state, step counter, diffusion configuration, scalar normalizers, action normalizers, and architecture configuration. Evaluation does not use teacher targets.
7. Prediction loss must reach the online shared force encoder and FiLM path and the prediction head. The action path must still receive gradients through its own tactile adapter. A detached `V_tilde` shared with the predictor would defeat the stated auxiliary training purpose.

An all-zero or nearly constant representation can minimize self-distillation targets in a degenerate setting. EMA/no-grad correctness and an overfit smoke test do not prove mitigation of modality collapse or real-robot success; those paper claims need experiments.

## 6. Predictor compatibility with the product group

The force target transforms with `Rf` and is invariant to `Rp`. If the implemented predictor is claimed to be equivariant, require

`P_psi(V_tilde_p @ Rp.T, V_tilde_f @ Rf.T) = P_psi(V_tilde_p, V_tilde_f) @ Rf.T`.

Do not use a single generic VN layer over the concatenated position and force vector channels; those channels transform under different rotations. This would violate equation (6) even though it might pass a common-rotation test with `Rp == Rf`.

The paper does not provide `P_psi`'s layer architecture or the precise cross-type interaction. A straightforward compatible implementation choice is to let force-type VN features provide output vectors while per-finger scalar invariants of the position-type field modulate those vectors. Such scalar gating can retain dependence on both parts of the shared state without identifying their coordinate frames. Document the particular invariant, gating MLP, channel dimensions, and horizon output construction as implementation choices. Do not attribute this architecture to the paper. Using invariants internally to form scalar coefficients does not replace the ordered vector state with an invariant pooled output.

The action expert is a conventional scalar model after flattening. The paper does not establish equivariance of the complete action policy, Cartesian arm actions, hand-joint commands, image encoder, or proprioceptive encoder under these synthetic typed rotations. The complete policy must not be advertised or tested as having an unprovided joint action transformation law.

## 7. Online lifecycle and deterministic prefix consistency

Recommended auditable interface:

1. `start_cycle(images_history, proprio_history, timestamp=..., initial_noise=None)` computes/caches slow tokens, pooled context, `Gamma`, and `[B,Ha,da]` Gaussian noise; it clears the previous cycle's fast state. It accepts deterministic noise injection for tests and reproducible replay.
2. `step(position_j, local_force_j)` validates five-finger inputs, appends exactly one new tactile frame, obtains causal `V_tilde[1:j]`, runs deterministic DDIM from the original noise slice `[1:j]`, and returns the newest action `[j]`. The public execution result is one action of shape `[B,da]`; a full prefix can be returned separately for diagnostics.
3. Reject a seventeenth fast action without a new cycle, or handle a documented visual-cache rollover policy. The paper specifies a 16-slot cycle and does not specify asynchronous overrun handling.
4. Teacher and predictor must never be called by the controller. No predicted tactile frame is appended to the observed tactile buffer.

Deterministic DDIM must use `eta=0`, fixed denoising step indices/schedule, the same denoiser, and the same initial-noise coordinates at every prefix extension. For diffusion step `k` to its previous selected step `s`, the standard epsilon-parameterized deterministic update is

`x0_hat = (x_k - sqrt(1-alpha_bar[k]) * epsilon_hat) / sqrt(alpha_bar[k])`

`x_s = sqrt(alpha_bar[s]) * x0_hat + sqrt(1-alpha_bar[s]) * epsilon_hat`,

with terminal clean step `alpha_bar = 1`. Training noise schedule and inference subsampling are implementation choices. Coordinate-wise optional clipping changes the sampler and must be documented; sequence-wide dynamic thresholding can break prefix consistency and should be avoided.

Required prefix property, within floating-point tolerance:

`sample(noise[:j], context, tactile[:j]) == sample(noise[:j+1], context, tactile[:j+1])[:j]`.

This follows by induction over the DDIM steps only if (a) every denoiser output at position `r` depends exclusively on available positions `<=r`; (b) old tactile states and FiLM/context remain unchanged; (c) original noise slots and schedule are identical; and (d) dropout and other stochastic evaluation behavior are disabled, with no prefix-length-dependent operations. Caching noise alone is insufficient. Reusing the previously denoised clean prefix as new initial noise implements a different process.

## 8. Parameters the paper does not provide

Record all values below under a clearly identified `implementation_choices` section of configuration/documentation. Do not call them recovered paper hyperparameters:

- `Cv`, VN layer count, hidden channels, equivariant nonlinearity/normalization details, attention heads, temporal receptive field, ordered-finger interactions, scalar labels/embeddings, initialization.
- Slow image backbone, pretrained weights, camera count beyond the experimental two-view setup, image resolution/crop/augmentations, image/proprioception fusion, tokenization, width, pooling, and proprioceptive dimensional/order/unit contract.
- FiLM MLP depth/width/activation and initialization. `[5,Cv]` output and no XYZ offset are fixed requirements.
- Action-expert width, depth, heads, diffusion-time embedding, scalar time/finger embeddings, dropout, internal tactile flatten/projection, exact slow cross-attention arrangement.
- Predictor architecture, optional position-invariant bridge into force outputs, horizon embedding/output parameterization, normalization, and whether all action origins are supervised.
- Training diffusion step count and beta schedule, inference step count/subsampling, action normalization/clipping, deterministic initial noise seeding policy. Epsilon prediction and deterministic DDIM are fixed requirements.
- `lambda`, feature-loss reductions, EMA momentum/schedule/update cadence, optimizer, learning rate/schedule, batch size, weight decay, number of epochs/steps, mixed precision, gradient accumulation/clipping, random seeds.
- Position/force scalar units and scale estimation; hand-joint action order/units; Cartesian increment frame and rotation representation; exact real hardware adapters, FK/calibration files, timestamp pairing, clock rates, missing-data policy, episode splits, target-window boundary behavior.

Important operational boundary: a clean training/inference repository can be implemented and verified without authentic raw datasets or hardware calibration. In that case it is an auditable executable method implementation, not a reproduction of the paper's trained weights or experimental performance. Synthetic examples must be labeled synthetic, and physical fingertip positions must be supplied by calibrated data or real FK rather than fabricated internally.

## 9. Minimum meaningful acceptance tests

These tests target distinct scientific requirements. Use randomized nonzero inputs and nontrivial independent rotations, multiple prefixes, and a tight float32 tolerance such as `atol=1e-5, rtol=1e-4` (adjust only with reported numeric evidence). Test model behavior rather than copying implementation formulas as the entire assertion.

1. **Independent typed equivariance and preserved shape.** Sample unrelated proper rotations `Rp` and `Rf` (`det=+1`). Verify each branch and shared FiLM state transform under their own type, with the other type unaffected, and five fingers retained. Also test the force predictor law if it is advertised as equivariant. Include zero/near-zero vectors to catch norm division errors. A common-rotation test alone is insufficient.
2. **Causal tactile encoder and predictor.** Encode full `T` and each prefix, then compare matching old outputs. Perturb frames after origin `r` strongly and verify `V_tilde[:r]` and predictions at origins `<=r` are unchanged. Verify gradients of old outputs to future inputs are zero where practical. This catches noncausal normalization/attention before the action mask.
3. **Denoiser future-action and future-touch isolation.** At fixed diffusion step and fixed context, perturb only noisy action slots `>r`, then only tactile blocks `>r`; predicted noise at slots `<=r` must remain unchanged. Check behavior when the newest visible tactile block is perturbed to ensure the pathway is not disconnected or trivially constant. Explicitly cover all five finger tokens sharing the same block visibility.
4. **Actual DDIM prefix equality and execution lifecycle.** Use fixed nonzero original noise and fixed slow cache. Compare each sampled prefix against the first slots of a longer/full sample. Test through the controller, count one returned executed action per newly arrived block, and verify cached old noise/context stay identical. A new cycle must reset state and recompute slow cache. Swapping/removing/raising inside predictor/teacher must leave inference working.
5. **Target indexing, causality, and boundaries.** With a controllable target encoder whose output identifies time/finger, assert origin `j` targets exactly `j+1,...,j+D`, and end-of-episode padding is masked instead of wrapped or used as valid labels. Perturb target-only future forces: targets/loss may change, but online `V_tilde`, action noise prediction, and predictor inputs must not. Perturb force after target time `t` and verify target at `t` remains fixed.
6. **EMA no-grad and real update.** Confirm initial exact equality of online/teacher state. Run a real nonzero training step and assert finite nonzero online gradients from `L_pred`, no teacher gradients, teacher absent from optimizer, and actual teacher parameter change that matches the chosen EMA recurrence. Verify checkpoint restore preserves teacher state and update counter. Merely checking `requires_grad=False` is insufficient.
7. **Noise objective and optimizer smoke.** Feed known clean action, chosen timestep, and controlled noise to check forward noising and epsilon target semantics. Run a small synthetic batch with real forward/backward/optimizer/EMA operations for each embodiment action size. Require finite losses/gradients, correct masks/dimensions, and successful save/load/inference. A falling training loss on a fixed tiny batch is a useful pipeline smoke check, not evidence of real-robot efficacy or anti-collapse success.

If compute time is constrained, tests 1–6 are the scientific acceptance gate; test 7 supplies the end-to-end usability check. Do not replace them with shape-only unit tests or a test suite that asserts only implementation constants.
