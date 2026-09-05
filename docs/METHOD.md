# Dex-EquiTact method

Dex-EquiTact combines a slow visual context with a fast tactile reaction state for dexterous manipulation. Two recent multi-view image and proprioception observations describe the task and scene. Within each visual cycle, incoming fingertip positions and forces update a structured tactile state, and a causal diffusion policy produces the next action. An auxiliary future-force objective trains the same state to retain information about contact evolution.

The method has five components: a slow context encoder, two causal vector-neuron tactile encoders, scalar FiLM modulation, a causal action denoiser, and a training-only future-force predictor. [Architecture](ARCHITECTURE.md) describes their code interfaces; [Data format](DATA_FORMAT.md) specifies the recorded inputs.

## Slow context and fast tactile observations

Let $\tau$ index visual cycles and $j=1,\ldots,H_a$ index fast steps within a cycle. The observation horizon is $H_o=2$, the action horizon is $H_a=16$, and the future-force prediction horizon is $D=8$. At cycle start, the slow encoder produces context tokens

$$
C_\tau=E_{\mathrm{slow}}\!\left(\{I,q\}_{\tau,H_o}\right),
$$

where $I$ contains the camera views and $q$ contains proprioception. This context is cached for the cycle. Each subsequent tactile block is observed before its corresponding action decision, so step $j$ can use only the tactile prefix through $j$.

Each block contains five ordered fingertip positions $P_{\tau,j}\in\mathbb R^{5\times3}$ and five calibrated resultant force vectors $F_{\tau,j}\in\mathbb R^{5\times3}$. The finger order is thumb, index, middle, ring, little. Positions are expressed in the wrist frame; each force remains in its sensor-local frame under the common calibrated local-axis convention. The two fields represent different geometric quantities and retain separate vector types throughout the tactile representation.

The slow encoder applies a three-layer trainable CNN to each image, retains a $2\times2$ grid of visual tokens, and adds camera, observation-time, and patch embeddings. An MLP embeds each proprioception observation. With two observations and two cameras, the result is 18 tokens: 16 visual tokens and two proprioception tokens. The context reaches the action denoiser directly and also controls tactile modulation.

## Typed causal tactile state

Separate encoders extract position and force features from the arrived prefix:

$$
\mathcal V^p_{\tau,j}=E_{\mathrm{pos}}(P_{\tau,1:j})_j,
\qquad
\mathcal V^f_{\tau,j}=E_{\mathrm{force}}(F_{\tau,1:j})_j.
$$

Each output has shape $[5,C_v,3]$, with one set of vector channels per finger. Stacking the two types gives

$$
\mathcal V_{\tau,j}
=\operatorname{stack}_{\mathrm{type}}
 (\mathcal V^p_{\tau,j},\mathcal V^f_{\tau,j})
\in\mathbb R^{5\times2\times C_v\times3}.
$$

The vector-neuron layers mix channels with weights shared over XYZ. Their nonlinear gates depend on vector norms, and their per-token RMS normalization uses an invariant scalar scale. Attention scores use vector dot products plus learned scalar time and finger-pair biases. These operations preserve vector transformation laws while retaining the five ordered finger outputs. All fingers at the same instant can interact; every attention layer blocks future time steps.

For independent rotations $R_p,R_f\in\mathrm{SO}(3)$, using row-vector coordinates,

$$
E_{\mathrm{pos}}(P R_p^\top)=E_{\mathrm{pos}}(P)R_p^\top,
\qquad
E_{\mathrm{force}}(F R_f^\top)=E_{\mathrm{force}}(F)R_f^\top.
$$

This is an $\mathrm{SO}(3)_p\times\mathrm{SO}(3)_f$ property of the typed tactile representation: each rotation is shared across the corresponding input field. It does not describe five independently rotated sensor frames. The action denoiser uses scalar token projections, so this representation property does not imply equivariance of the complete image-to-action policy.

## Context-dependent scalar FiLM

Visual context adapts the tactile state through scalar channel gains. Mean pooling the cached context and applying an MLP gives $\Gamma_\tau\in\mathbb R^{5\times C_v}$. For finger $i$, type $t$, and channel $c$,

$$
\widetilde{\mathcal V}_{\tau,j,i,t,c}
=(1+\Gamma_{\tau,i,c})\mathcal V_{\tau,j,i,t,c}.
$$

The gains are shared over time, vector type, and XYZ within the cycle. Scalar multiplication preserves each type's rotation law without adding a spatial offset. Both the action denoiser and future-force predictor consume this same modulated vector state. Flattening the type, channel, and XYZ axes occurs inside the action denoiser's per-finger token projection.

## Causal diffusion actions

The action sequence is $A_\tau\in\mathbb R^{H_a\times d_a}$. A WUJI policy uses $d_a=26$, comprising six arm increments and 20 hand targets; a Sharpa policy uses $d_a=28$, with six arm increments and 22 hand targets. Each embodiment has its own policy.

The denoiser embeds noisy actions, diffusion time, and absolute fast-step indices. Each block applies causal action self-attention, cross-attention to cached slow tokens and tactile tokens, and a feedforward layer. Action query $r$ may attend to action or tactile time $u$ exactly when

$$
M_{r,u}=\begin{cases}
0,&u\le r,\\
-\infty,&u>r.
\end{cases}
$$

Every action query can see all cached slow tokens and all five fingers at each allowed tactile time. The conditioning memory remains read-only during denoising. Per-token normalization and fixed time indices keep the same computation valid as the prefix grows.

### Training: predict injected noise

For sampled Gaussian noise $\epsilon$ and diffusion level $k$, define $\bar\alpha_k=\prod_{s=0}^{k}(1-\beta_s)$. The forward process produces

$$
A_\tau^{(k)}=\sqrt{\bar\alpha_k}\,A_\tau
+\sqrt{1-\bar\alpha_k}\,\epsilon.
$$

The denoiser $D_\theta(A_\tau^{(k)},k,C_\tau,\widetilde{\mathcal V}_\tau)$ predicts $\epsilon$. Its output is a noise estimate with the action shape. Causal masking permits full 16-step training while preserving each step's observation boundary.

### Inference: sample the arrived prefix

At cycle start, the controller draws and caches Gaussian noise $Z_\tau\in\mathbb R^{H_a\times d_a}$. When tactile step $j$ arrives, deterministic DDIM with $\eta=0$ samples

$$
\widehat A_{\tau,1:j}
=\operatorname{DDIM}\!\left(
Z_{\tau,1:j},C_\tau,\widetilde{\mathcal V}_{\tau,1:j}
\right).
$$

For a noisy sample $x_k$, DDIM first estimates the clean action as

$$
\widehat x_0=
\frac{x_k-\sqrt{1-\bar\alpha_k}\,D_\theta(x_k,k,C_\tau,\widetilde{\mathcal V})}
 {\sqrt{\bar\alpha_k}}.
$$

It then moves to the next selected diffusion level using this clean estimate and the predicted noise. The final update returns $\widehat x_0$. The sampler uses the original cached noise prefix for every tactile arrival, with fixed context and deterministic layers. Together with complete causal masking, this keeps earlier sampled actions consistent as more tactile observations arrive, up to floating-point roundoff.

### Execution: return the newest action

The controller returns only

$$
a^{\mathrm{exec}}_{\tau,j}=\widehat A_{\tau,1:j}[j].
$$

Earlier prefix slots are recomputed to obtain the current sample, but their actions are not returned again. A new visual cycle refreshes context and initial noise and clears the tactile prefix. The controller emits at most 16 actions per cycle.

## Future-force supervision

The auxiliary predictor trains the shared reaction state to describe upcoming contact. At each origin $j$, it predicts the next eight force-type latent fields:

$$
\widehat{\mathcal F}_{\tau,j,1:D}
=P_\psi(\widetilde{\mathcal V}_{\tau,j})
\in\mathbb R^{D\times5\times C_v\times3}.
$$

The head projects force-vector channels into horizon-specific output channels. Position information supplies scalar gates through $\log(1+\|\widetilde{\mathcal V}^p\|^2)$, an MLP, and $1+\tanh$. Thus position rotations leave its predictions unchanged, while force rotations rotate the predicted force vectors. The head takes the shared tactile state directly, without an action input or recursive predicted-state feedback.

Targets come from a frozen exponential-moving-average copy of the force encoder:

$$
\mathcal F^*_{\tau,j,d}
=\operatorname{sg}\!\left[
\overline E_{\mathrm{force}}(F_{\tau,1:j+d})_{j+d}
\right],\qquad d=1,\ldots,8.
$$

The teacher is force-only, causal, and unmodulated by FiLM. Training supplies 16 current force frames plus eight continuation frames from the same episode, yielding a target for all 128 origin/horizon pairs. Those eight extra frames are used only by the teacher. The teacher starts as an exact copy of the online force encoder and updates once after each successful optimizer step:

$$
\overline\theta_f\leftarrow m\overline\theta_f+(1-m)\theta_f.
$$

Both the predictor and teacher are used during training. Reactive inference uses the slow encoder, tactile encoder, FiLM, and action denoiser.

## Joint objective and default configuration

The action loss sums squared noise error across action coordinates, then averages over batch and action time. The prediction loss sums squared force-latent error across finger, channel, and XYZ axes, then averages over batch, origin, and future horizon:

$$
\mathcal L_{\mathrm{act}}=\frac{1}{BH_a}
\sum_{b,j}\|\widehat\epsilon_{b,j}-\epsilon_{b,j}\|_2^2,
\qquad
\mathcal L_{\mathrm{pred}}=\frac{1}{BH_aD}
\sum_{b,j,d}\|\widehat{\mathcal F}_{b,j,d}-\mathcal F^*_{b,j,d}\|_F^2,
$$

$$
\mathcal L=\mathcal L_{\mathrm{act}}+\lambda\mathcal L_{\mathrm{pred}}.
$$

Gradients update the participating online modules; teacher targets are detached and teacher parameters are excluded from optimization. Full within-episode windows avoid padding in these reductions.

The current [WUJI](../configs/wuji.yaml) and [Sharpa](../configs/sharpa.yaml) configurations use the following defaults:

| Setting | Value |
| --- | --- |
| Tactile vector channels / layers / attention heads | 32 / 2 / 4 per stream |
| Context and action width | 128 |
| Action layers / attention heads | 2 / 4 |
| Diffusion schedule | 1,000 levels; linear beta from 0.0001 to 0.02 |
| DDIM sampling | 10 rounded, evenly spaced levels; deterministic updates |
| Prediction weight $\lambda$ / EMA decay $m$ | 1.0 / 0.99 |
| Optimizer | AdamW; learning rate 0.0001; weight decay 0.000001 |
| Gradient norm clipping | 1.0 |
| Training batch size / seed | 8 / 0 |
| Validation fraction / slow observation stride | 0.2 / 16 fast-grid samples |

Images enter as RGB floats in $[0,1]$; the loader applies no resize or augmentation. Position and force normalization use separate scalar scales shared across XYZ and fingers. Proprioception and actions use training-only per-coordinate mean and standard deviation. The training output saves the selected configuration and normalization state for evaluation, resume, and deployment.
