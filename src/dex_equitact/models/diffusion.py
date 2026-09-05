"""Causal epsilon prediction and deterministic growing-prefix DDIM.

Equations (11)--(13) of Dex-EquiTact specify visibility, epsilon prediction,
and deterministic sampling. Widths, layer counts, sinusoidal diffusion-time
features, learned absolute position embeddings, linear beta schedules, and
inference step counts below are explicit implementation choices; the paper
does not provide their values. This conventional scalar action expert does
not claim the typed rotational equivariance of its incoming tactile field.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _CausalActionBlock(nn.Module):
    """Update actions while keeping slow and tactile memory read-only."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width),
        )

    def forward(
        self, actions: Tensor, memory: Tensor, self_mask: Tensor, cross_mask: Tensor,
    ) -> Tensor:
        normalized = self.self_norm(actions)
        update, _ = self.self_attention(
            normalized, normalized, normalized,
            attn_mask=self_mask, need_weights=False,
        )
        actions = actions + update
        update, _ = self.cross_attention(
            self.cross_norm(actions), memory, memory,
            attn_mask=cross_mask, need_weights=False,
        )
        actions = actions + update
        return actions + self.feedforward(self.feedforward_norm(actions))


class ActionDenoiser(nn.Module):
    """Predict injected action noise from a causal shared tactile vector field.

    Inputs are ``[B,T,A]`` noisy actions, ``[B]`` integer diffusion timesteps,
    cached slow tokens ``[B,N,E]``, and the shared FiLM-modulated vector field
    ``[B,T,5,2,C,3]``. Only this expert flattens each finger's typed vectors.

    Every action sees all slow tokens, action slots up to its own index, and
    all five fingers of tactile blocks up to its own index. No memory token
    is updated using actions. Absolute position indices and per-token norms
    are independent of prefix length. All attention dropout is zero in both
    training and evaluation.

    Apart from the five-finger typed input and the paper's 16-step horizon,
    the architecture defaults are implementation choices, not recovered
    paper hyperparameters. Any positive action dimension is accepted; the
    paper's separate embodiments use 26 and 28.
    """

    def __init__(
        self,
        action_dim: int,
        channels: int,
        model_dim: int = 128,
        layers: int = 2,
        heads: int = 4,
        max_steps: int = 16,
        context_dim: int = 128,
    ) -> None:
        super().__init__()
        if any(value < 1 for value in (
            action_dim, channels, model_dim, layers, heads, max_steps, context_dim,
        )):
            raise ValueError("All architecture dimensions and layer counts must be positive.")
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads.")
        self.action_dim = action_dim
        self.channels = channels
        self.model_dim = model_dim
        self.max_steps = max_steps
        self.context_dim = context_dim
        self.action_projection = nn.Linear(action_dim, model_dim)
        self.tactile_projection = nn.Linear(2 * channels * 3, model_dim)
        self.slow_projection = nn.Linear(context_dim, model_dim)
        self.action_positions = nn.Embedding(max_steps, model_dim)
        self.tactile_positions = nn.Embedding(max_steps, model_dim)
        self.finger_positions = nn.Embedding(5, model_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim),
        )
        self.memory_norm = nn.LayerNorm(model_dim)
        self.blocks = nn.ModuleList([
            _CausalActionBlock(model_dim, heads) for _ in range(layers)
        ])
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, action_dim)

    def _time_features(self, timesteps: Tensor) -> Tensor:
        half = self.model_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        phases = timesteps[:, None].float() * frequencies[None]
        features = torch.cat((phases.sin(), phases.cos()), dim=-1)
        if self.model_dim % 2:
            features = F.pad(features, (0, 1))
        return self.time_mlp(features.to(dtype=self.action_projection.weight.dtype))

    def forward(
        self,
        noisy_actions: Tensor,
        timesteps: Tensor,
        slow_tokens: Tensor,
        tactile_field: Tensor,
    ) -> Tensor:
        if noisy_actions.ndim != 3 or noisy_actions.shape[-1] != self.action_dim:
            raise ValueError(f"noisy_actions must have shape [B,T,{self.action_dim}].")
        batch, length, _ = noisy_actions.shape
        if batch < 1 or not 1 <= length <= self.max_steps:
            raise ValueError("The action batch must be nonempty and T must be within max_steps.")
        if timesteps.shape != (batch,) or timesteps.dtype != torch.long:
            raise ValueError("timesteps must be a LongTensor of shape [B].")
        if torch.any(timesteps < 0):
            raise ValueError("timesteps must be nonnegative.")
        expected_tactile = (batch, length, 5, 2, self.channels, 3)
        if tuple(tactile_field.shape) != expected_tactile:
            raise ValueError(f"tactile_field must have shape {expected_tactile}.")
        if (
            slow_tokens.ndim != 3
            or slow_tokens.shape[0] != batch
            or slow_tokens.shape[1] < 1
            or slow_tokens.shape[2] != self.context_dim
        ):
            raise ValueError(f"slow_tokens must have shape [B,N,{self.context_dim}], N >= 1.")

        positions = torch.arange(length, device=noisy_actions.device)
        actions = self.action_projection(noisy_actions)
        actions = actions + self.action_positions(positions)[None]
        actions = actions + self._time_features(timesteps)[:, None]

        # The canonical field remains vector-valued outside this interface.
        tactile_tokens = self.tactile_projection(tactile_field.flatten(start_dim=3))
        tactile_tokens = tactile_tokens + self.tactile_positions(positions)[None, :, None]
        finger_ids = torch.arange(5, device=noisy_actions.device)
        tactile_tokens = tactile_tokens + self.finger_positions(finger_ids)[None, None]
        tactile_tokens = tactile_tokens.reshape(batch, 5 * length, self.model_dim)
        memory = torch.cat((self.slow_projection(slow_tokens), tactile_tokens), dim=1)
        memory = self.memory_norm(memory)

        # nn.MultiheadAttention boolean masks use True to BLOCK a key.
        # Repeating each tactile column five times gives every current finger
        # identical temporal visibility; no within-frame causal ordering.
        self_mask = positions[None, :] > positions[:, None]
        tactile_mask = self_mask.repeat_interleave(5, dim=1)
        slow_mask = torch.zeros(
            length, slow_tokens.shape[1], device=noisy_actions.device, dtype=torch.bool,
        )
        cross_mask = torch.cat((slow_mask, tactile_mask), dim=1)
        for block in self.blocks:
            actions = block(actions, memory, self_mask, cross_mask)
        return self.output_projection(self.output_norm(actions))


class DiffusionSchedule(nn.Module):
    """Epsilon-prediction forward process and deterministic DDIM (eta=0).

    The linear beta schedule, 1000 training steps, and default ten inference
    steps are implementation choices. Selected inference indices are rounded
    evenly spaced indices from ``train_steps - 1`` to zero. One inference
    step starts at ``train_steps - 1`` and directly returns a clean estimate.
    There is no stochastic sampling step, clipping, or dynamic thresholding.
    """

    def __init__(
        self, train_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
    ) -> None:
        super().__init__()
        if not isinstance(train_steps, int) or train_steps < 1:
            raise ValueError("train_steps must be a positive integer.")
        if not 0 < beta_start <= beta_end < 1:
            raise ValueError("The beta schedule must satisfy 0 < beta_start <= beta_end < 1.")
        self.train_steps = train_steps
        betas = torch.linspace(beta_start, beta_end, train_steps, dtype=torch.float64)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alpha_bar", (1.0 - betas).cumprod(dim=0).float())

    def add_noise(self, actions: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """Apply equation (12); the supervised target remains ``noise``."""
        if actions.ndim != 3 or actions.shape != noise.shape:
            raise ValueError("actions and noise must share shape [B,T,A].")
        if timesteps.shape != (actions.shape[0],) or timesteps.dtype != torch.long:
            raise ValueError("timesteps must be a LongTensor of shape [B].")
        if torch.any(timesteps < 0) or torch.any(timesteps >= self.train_steps):
            raise ValueError("timesteps must lie in [0, train_steps).")
        alpha = self.alpha_bar.to(actions)[timesteps.to(actions.device)][:, None, None]
        return alpha.sqrt() * actions + (1.0 - alpha).sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        denoiser: nn.Module,
        initial_noise: Tensor,
        slow_tokens: Tensor,
        tactile_field: Tensor,
        inference_steps: int = 10,
    ) -> Tensor:
        """Return a clean prefix without altering the cached initial noise.

        For growing-prefix control, the caller must pass the same original
        noise slots and cached slow context at every extension, together with
        only the arrived tactile prefix. This method returns all requested
        slots; the controller executes only the newest one. ``denoiser`` must
        be deterministic; ActionDenoiser has no dropout in either mode.
        """
        if not isinstance(inference_steps, int) or not 1 <= inference_steps <= self.train_steps:
            raise ValueError("inference_steps must be an integer in [1, train_steps].")
        if initial_noise.ndim != 3 or any(size < 1 for size in initial_noise.shape):
            raise ValueError("initial_noise must have nonempty shape [B,T,A].")
        indices = torch.linspace(
            self.train_steps - 1, 0, inference_steps, dtype=torch.float64,
        ).round().long().tolist()
        sample = initial_noise.clone()
        for index, timestep in enumerate(indices):
            batch_times = torch.full(
                (sample.shape[0],), timestep, device=sample.device, dtype=torch.long,
            )
            predicted_noise = denoiser(sample, batch_times, slow_tokens, tactile_field)
            if predicted_noise.shape != sample.shape:
                raise ValueError("The denoiser must predict noise with the action shape.")
            alpha = self.alpha_bar[timestep].to(sample)
            clean = (sample - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
            if index + 1 == len(indices):
                # The terminal clean level has alpha_prev = 1, not alpha_bar[0].
                sample = clean
            else:
                previous_alpha = self.alpha_bar[indices[index + 1]].to(sample)
                sample = (
                    previous_alpha.sqrt() * clean
                    + (1.0 - previous_alpha).sqrt() * predicted_noise
                )
        return sample
