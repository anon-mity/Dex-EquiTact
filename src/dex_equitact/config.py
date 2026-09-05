"""Paper constants and explicitly configurable engineering defaults."""
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PolicyConfig:
    action_dim: int = 26
    proprio_dim: int = 26
    num_views: int = 2
    observation_horizon: int = 2
    action_horizon: int = 16
    prediction_horizon: int = 8
    vector_channels: int = 32
    vector_layers: int = 2
    vector_heads: int = 4
    model_dim: int = 128
    action_layers: int = 2
    action_heads: int = 4
    train_diffusion_steps: int = 1000
    inference_steps: int = 10
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_weight: float = 1.0
    ema_decay: float = 0.99

    def __post_init__(self):
        if self.action_dim not in (26, 28):
            raise ValueError('Paper embodiments require action_dim 26 (WUJI) or 28 (Sharpa).')
        if (self.observation_horizon, self.action_horizon, self.prediction_horizon) != (2, 16, 8):
            raise ValueError('Paper horizons are observation=2, action=16, prediction=8.')
        for name in ('proprio_dim', 'num_views', 'vector_channels', 'vector_layers',
                     'vector_heads', 'model_dim', 'action_layers', 'action_heads',
                     'train_diffusion_steps', 'inference_steps'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.model_dim % self.action_heads or self.vector_channels % self.vector_heads:
            raise ValueError('Channels/model dimension must be divisible by the respective heads.')
        if self.model_dim < 8 or self.model_dim % 4:
            raise ValueError('model_dim must be a multiple of four and at least eight.')
        if not 1 <= self.inference_steps <= self.train_diffusion_steps:
            raise ValueError('Invalid inference_steps')
        if not 0 < self.beta_start < self.beta_end < 1:
            raise ValueError('Require 0 < beta_start < beta_end < 1')
        if not 0 <= self.ema_decay < 1 or not self.prediction_weight > 0:
            raise ValueError('Require 0 <= ema_decay < 1 and prediction_weight > 0')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_yaml(cls, path):
        with Path(path).open() as f:
            values = yaml.safe_load(f)
        if not isinstance(values, dict):
            raise ValueError('Policy configuration must be a YAML mapping')
        return cls(**values)
