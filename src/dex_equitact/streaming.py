"""Hardware-independent visual-cycle cache; returns only the newest action."""
import math

import torch


class ReactiveController:
    def __init__(self, policy, normalizer=None):
        self.policy = policy
        self.normalizer = normalizer
        self.reset()

    def reset(self):
        self.context = self.gains = self.initial_noise = None
        self.positions, self.forces = [], []
        self.last_timestamp = None
        self.cycle_timestamp = None

    @torch.no_grad()
    def start_cycle(self, images, proprio, *, timestamp, initial_noise=None, generator=None):
        """Slow observations are already available at timestamp. No sensor I/O.

        With a normalizer, supply physical-unit proprio/tactile data and get
        physical-unit actions. Without one all nonimage values are normalized.
        Images are floating RGB [0,1]. New cycles explicitly clear tactile history.
        """
        if self.policy.training:
            raise RuntimeError('Call policy.eval() before starting reactive inference')
        if not math.isfinite(timestamp):
            raise ValueError('timestamp must be finite')
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError('New cycle timestamp precedes the last tactile observation')
        p = self.normalizer.normalize_proprio(proprio) if self.normalizer else proprio
        if not torch.isfinite(p).all():
            raise ValueError('proprio must be finite')
        context = self.policy.slow_encoder(images, p)
        c = self.policy.config
        shape = (images.shape[0], c.action_horizon, c.action_dim)
        if initial_noise is None:
            initial_noise = torch.randn(shape, device=context.device, dtype=context.dtype,
                                        generator=generator)
        if initial_noise.shape != shape or not torch.isfinite(initial_noise).all():
            raise ValueError('initial_noise must be finite [B,16,action_dim]')
        previous_tactile_timestamp = self.last_timestamp
        self.reset()
        self.context = context.detach().clone()
        self.gains = self.policy.film.gains(context).detach().clone()
        self.initial_noise = initial_noise.to(context).detach().clone()
        self.last_timestamp = previous_tactile_timestamp
        self.cycle_timestamp = timestamp

    @torch.no_grad()
    def step(self, positions, forces, *, timestamp):
        if self.context is None:
            raise RuntimeError('start_cycle must be called before step')
        if self.policy.training:
            raise RuntimeError('Reactive inference requires policy.eval()')
        if (not math.isfinite(timestamp) or timestamp < self.cycle_timestamp or
                (self.last_timestamp is not None and timestamp <= self.last_timestamp)):
            raise ValueError('Each tactile timestamp must be finite and strictly increasing')
        if len(self.forces) >= self.policy.config.action_horizon:
            raise RuntimeError('Visual cycle exhausted: start a new cycle before more actions')
        shape = (self.context.shape[0], 5, 3)
        if positions.shape != shape or forces.shape != shape:
            raise ValueError('Each tactile block must be [B,5,3]')
        if not torch.isfinite(positions).all() or not torch.isfinite(forces).all():
            raise ValueError('Tactile blocks must be finite')
        if self.normalizer:
            positions, forces = self.normalizer.normalize_tactile(positions, forces)
        p = torch.stack(self.positions + [positions.to(self.context).detach().clone()], dim=1)
        f = torch.stack(self.forces + [forces.to(self.context).detach().clone()], dim=1)
        field = self.policy.film(self.policy.tactile(p, f), self.gains)
        sampled = self.policy.diffusion.sample(self.policy.denoiser,
                    self.initial_noise[:, :f.shape[1]], self.context, field,
                    self.policy.config.inference_steps)
        if not torch.isfinite(sampled).all():
            raise RuntimeError('Denoiser generated a nonfinite action')
        newest = sampled[:, -1]
        if self.normalizer:
            newest = self.normalizer.denormalize_actions(newest)
        # Commit the arriving block only after successful inference.
        self.positions.append(p[:, -1])
        self.forces.append(f[:, -1])
        self.last_timestamp = timestamp
        return newest
