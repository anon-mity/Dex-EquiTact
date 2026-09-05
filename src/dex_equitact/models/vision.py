"""Encode multiview RGB and proprioception into cached slow-context tokens.

The image CNN produces a 2x2 spatial token grid for each camera observation.
Proprioceptive tokens and camera/time/patch embeddings complete the slow context.
Image normalization operates per image, preserving fast-sequence causality.
"""
import torch
from torch import nn


class SlowContextEncoder(nn.Module):
    def __init__(self, proprio_dim, num_views=2, model_dim=128, observation_horizon=2):
        super().__init__()
        self.proprio_dim, self.num_views = proprio_dim, num_views
        self.observation_horizon = observation_horizon
        width = model_dim // 2
        self.cnn = nn.Sequential(
            nn.Conv2d(3, width, 5, stride=2, padding=2), nn.GroupNorm(2, width), nn.SiLU(),
            nn.Conv2d(width, model_dim, 3, stride=2, padding=1),
            nn.GroupNorm(4, model_dim), nn.SiLU(),
            nn.Conv2d(model_dim, model_dim, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)))
        self.proprio = nn.Sequential(nn.Linear(proprio_dim, model_dim), nn.SiLU(),
                                    nn.Linear(model_dim, model_dim))
        self.camera_embedding = nn.Parameter(torch.randn(num_views, model_dim) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(observation_horizon, model_dim) * 0.02)
        self.patch_embedding = nn.Parameter(torch.randn(4, model_dim) * 0.02)
        self.proprio_embedding = nn.Parameter(torch.randn(model_dim) * 0.02)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, images, proprio):
        if images.ndim != 6:
            raise ValueError('images must be [B,Ho,V,3,H,W] floats in [0,1]')
        b, ho, v, c, h, w = images.shape
        if (ho, v, c) != (self.observation_horizon, self.num_views, 3) or min(h, w) < 8:
            raise ValueError('Invalid image horizon, camera count, channels or resolution')
        if proprio.shape != (b, ho, self.proprio_dim):
            raise ValueError('Invalid proprio shape')
        if not images.is_floating_point() or not torch.isfinite(images).all() or (
                images.min() < 0 or images.max() > 1):
            raise ValueError('images must contain finite floats in [0,1]')
        visual = self.cnn(images.reshape(b * ho * v, c, h, w)).flatten(2).transpose(1, 2)
        visual = visual.reshape(b, ho, v, 4, -1)
        visual = (visual + self.camera_embedding[None, None, :, None, :]
                  + self.time_embedding[None, :, None, None, :]
                  + self.patch_embedding[None, None, None, :, :])
        state = self.proprio(proprio) + self.time_embedding + self.proprio_embedding
        return self.norm(torch.cat([visual.reshape(b, -1, visual.shape[-1]), state], dim=1))
