"""Shared typed tactile state, epsilon prediction, and training-only force target."""
from copy import deepcopy

import torch
from torch import nn

from .config import PolicyConfig
from .models.vision import SlowContextEncoder
from .models.tactile import TypedTactileEncoder, ScalarFiLM, FutureForcePredictor
from .models.diffusion import ActionDenoiser, DiffusionSchedule


class DexEquiTactPolicy(nn.Module):
    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = c = config
        self.slow_encoder = SlowContextEncoder(c.proprio_dim, c.num_views, c.model_dim,
                                              c.observation_horizon)
        self.tactile = TypedTactileEncoder(c.vector_channels, c.vector_layers, c.vector_heads,
                                          max_steps=c.action_horizon + c.prediction_horizon)
        self.film = ScalarFiLM(c.model_dim, c.vector_channels)
        self.denoiser = ActionDenoiser(c.action_dim, c.vector_channels, c.model_dim,
                                      c.action_layers, c.action_heads, c.action_horizon,
                                      context_dim=c.model_dim)
        self.diffusion = DiffusionSchedule(c.train_diffusion_steps, c.beta_start, c.beta_end)
        self.predictor = FutureForcePredictor(c.vector_channels, c.prediction_horizon)
        self.force_ema = deepcopy(self.tactile.force_encoder).requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.force_ema.eval()
        return self

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def update_ema(self, decay=None):
        """Call once AFTER each successful optimizer.step(), never per microbatch."""
        m = self.config.ema_decay if decay is None else decay
        if not 0 <= m < 1:
            raise ValueError('EMA decay must be in [0,1)')
        online = dict(self.tactile.force_encoder.named_parameters())
        for name, teacher in self.force_ema.named_parameters():
            teacher.lerp_(online[name], 1 - m)
        online_buffers = dict(self.tactile.force_encoder.named_buffers())
        for name, teacher in self.force_ema.named_buffers():
            teacher.copy_(online_buffers[name])

    @torch.no_grad()
    def future_targets(self, forces, future_forces):
        c = self.config
        if forces.ndim != 4 or forces.shape[1:] != (c.action_horizon, 5, 3):
            raise ValueError('forces must contain all 16 current frames')
        if future_forces.shape != (forces.shape[0], c.prediction_horizon, 5, 3):
            raise ValueError('future_forces must contain exactly 8 target-only frames')
        encoded = self.force_ema(torch.cat([forces, future_forces], dim=1))
        # Zero-based origin j targets j+1,...,j+D; last origin reaches sample 23.
        indices = (torch.arange(c.action_horizon, device=forces.device)[:, None]
                   + torch.arange(1, c.prediction_horizon + 1, device=forces.device)[None, :])
        return encoded[:, indices].detach()

    def loss(self, batch, *, timesteps=None, noise=None):
        """Inputs normalized externally. Frobenius sums, averaged over valid slots.

        L_act sums action coordinates and averages batch/time. L_pred sums
        finger/channel/XYZ and averages batch/origin/future horizon. No padding.
        """
        c = self.config
        actions, forces = batch['actions'], batch['forces']
        b = actions.shape[0]
        if actions.shape != (b, c.action_horizon, c.action_dim):
            raise ValueError('Invalid actions shape')
        if batch['positions'].shape != (b, c.action_horizon, 5, 3):
            raise ValueError('Invalid positions shape')
        for name, value in batch.items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite batch field: {name}')
        targets = self.future_targets(forces, batch['future_forces'])
        context = self.slow_encoder(batch['images'], batch['proprio'])
        field = self.film(self.tactile(batch['positions'], forces), self.film.gains(context))
        if timesteps is None:
            timesteps = torch.randint(c.train_diffusion_steps, (b,), device=actions.device)
        if noise is None:
            noise = torch.randn_like(actions)
        noisy = self.diffusion.add_noise(actions, noise, timesteps)
        predicted_noise = self.denoiser(noisy, timesteps, context, field)
        predicted_force = self.predictor(field)
        action_loss = (predicted_noise - noise).square().sum(dim=-1).mean()
        prediction_loss = (predicted_force - targets).square().sum(dim=(-3, -2, -1)).mean()
        return {'loss': action_loss + c.prediction_weight * prediction_loss,
                'action_loss': action_loss, 'prediction_loss': prediction_loss}

    def forward(self, batch):
        return self.loss(batch)
