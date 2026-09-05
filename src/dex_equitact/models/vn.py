"""Causal vector-neuron building blocks using PyTorch only.

The paper fixes equivariance, causality and ordered fingers, but not these
layer details. This implementation chooses scalar gated vector MLPs, vector
RMS normalization, and learned scalar relative-time / finger-pair attention
biases. Learned maps never mix XYZ coordinates or add fixed spatial vectors.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class VectorLinear(nn.Module):
    """Mix vector channels with the same weights on each spatial coordinate."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_channels, input_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...ci,oc->...oi", vectors, self.weight)


class VectorRMSNorm(nn.Module):
    """Normalize each token by one invariant RMS across its vector channels."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        mean_squared_norm = vectors.square().sum(-1).mean(-1, keepdim=True)
        scale = (mean_squared_norm + self.eps).rsqrt().unsqueeze(-1)
        return vectors * scale * self.gain[:, None]


class VectorMLP(nn.Module):
    """A nonlinear vector map gated by rotation-invariant scalar features."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = 2 * channels
        self.input = VectorLinear(channels, hidden)
        self.gate = nn.Linear(hidden, hidden)
        self.output = VectorLinear(hidden, channels)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        hidden = self.input(vectors)
        invariants = torch.log1p(hidden.square().sum(-1))
        gate = torch.sigmoid(self.gate(invariants))
        return self.output(hidden * gate.unsqueeze(-1))


class CausalVectorAttention(nn.Module):
    """Attention over ordered time/finger tokens with invariant scalar scores."""

    def __init__(self, channels: int, heads: int, max_steps: int):
        super().__init__()
        self.heads = heads
        self.head_channels = channels // heads
        self.query = VectorLinear(channels, channels)
        self.key = VectorLinear(channels, channels)
        self.value = VectorLinear(channels, channels)
        self.output = VectorLinear(channels, channels)
        self.time_bias = nn.Parameter(torch.zeros(heads, max_steps))
        self.finger_bias = nn.Parameter(torch.empty(heads, 5, 5))
        # Nonconstant scalar bias encodes fixed finger identity at initialization.
        nn.init.normal_(self.finger_bias, std=0.02)

    def _heads(self, vectors: torch.Tensor) -> torch.Tensor:
        batch, tokens = vectors.shape[:2]
        return vectors.reshape(batch, tokens, self.heads, self.head_channels * 3).transpose(1, 2)

    def forward(self, vectors: torch.Tensor, times: torch.Tensor, fingers: torch.Tensor) -> torch.Tensor:
        query = self._heads(self.query(vectors))
        key = self._heads(self.key(vectors))
        value = self._heads(self.value(vectors))
        scores = query @ key.transpose(-1, -2) / math.sqrt(3 * self.head_channels)
        lag = times[:, None] - times[None, :]
        bias = self.time_bias[:, lag.clamp_min(0)]
        bias = bias + self.finger_bias[:, fingers[:, None], fingers[None, :]]
        scores = (scores + bias).masked_fill(lag[None, None] < 0, -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        attended = (attention @ value).transpose(1, 2)
        attended = attended.reshape(*vectors.shape)
        return self.output(attended)


class CausalVectorBlock(nn.Module):
    def __init__(self, channels: int, heads: int, max_steps: int):
        super().__init__()
        self.attention_norm = VectorRMSNorm(channels)
        self.attention = CausalVectorAttention(channels, heads, max_steps)
        self.mlp_norm = VectorRMSNorm(channels)
        self.mlp = VectorMLP(channels)

    def forward(self, vectors: torch.Tensor, times: torch.Tensor, fingers: torch.Tensor) -> torch.Tensor:
        vectors = vectors + self.attention(self.attention_norm(vectors), times, fingers)
        return vectors + self.mlp(self.mlp_norm(vectors))


class CausalVNEncoder(nn.Module):
    """Map [B,T,5,3] to ordered SO(3) carriers [B,T,5,C,3].

    All five fingers within a time step may interact; future time steps are
    masked in every layer. No cross-finger pooling or invariant output readout
    occurs. Prefixes use the same relative-time and finger biases as the full
    sequence. This module contains no stochastic layers.
    """

    def __init__(self, channels: int, layers: int, heads: int, max_steps: int = 24):
        super().__init__()
        if channels <= 0 or heads <= 0 or channels % heads:
            raise ValueError("channels must be positive and divisible by positive heads")
        if layers <= 0 or max_steps <= 0:
            raise ValueError("layers and max_steps must be positive")
        self.channels = channels
        self.max_steps = max_steps
        self.input = VectorLinear(1, channels)
        self.blocks = nn.ModuleList(CausalVectorBlock(channels, heads, max_steps) for _ in range(layers))

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        if sample.ndim != 4 or sample.shape[-2:] != (5, 3):
            raise ValueError("CausalVNEncoder expects [B,T,5,3], with exactly 5 ordered fingers")
        batch, steps = sample.shape[:2]
        if not 1 <= steps <= self.max_steps:
            raise ValueError(f"T must be between 1 and max_steps={self.max_steps}; got {steps}")
        vectors = self.input(sample.unsqueeze(-2)).reshape(batch, steps * 5, self.channels, 3)
        times = torch.arange(steps, device=sample.device).repeat_interleave(5)
        fingers = torch.arange(5, device=sample.device).repeat(steps)
        for block in self.blocks:
            vectors = block(vectors, times, fingers)
        return vectors.reshape(batch, steps, 5, self.channels, 3)
