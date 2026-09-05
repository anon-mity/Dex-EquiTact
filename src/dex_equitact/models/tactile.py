"""Typed tactile carriers, scalar FiLM and equivariant future-force readout.

Vector type 0 is wrist-frame position; type 1 is local-sensor force. The two
streams transform under independent SO(3) actions. Scalar FiLM gains are shared
across the type and XYZ axes. The future-force predictor uses position-norm
scalar gates and force-vector channel projections to preserve the force-type
rotation law while conditioning on both parts of the shared tactile state.
"""

from __future__ import annotations

import torch
from torch import nn

from .vn import CausalVNEncoder, VectorLinear


def _validate_field(field: torch.Tensor, channels: int) -> None:
    if field.ndim != 6 or field.shape[-4:] != (5, 2, channels, 3):
        raise ValueError(f"Expected typed field [B,T,5,2,{channels},3]")


class TypedTactileEncoder(nn.Module):
    """Independently encode position and force, retaining both vector types."""

    def __init__(self, channels: int, layers: int, heads: int, max_steps: int = 24):
        super().__init__()
        self.position_encoder = CausalVNEncoder(channels, layers, heads, max_steps)
        self.force_encoder = CausalVNEncoder(channels, layers, heads, max_steps)

    def forward(self, positions: torch.Tensor, forces: torch.Tensor) -> torch.Tensor:
        if positions.shape != forces.shape:
            raise ValueError("Position and force inputs must have matching [B,T,5,3] shapes")
        position_vectors = self.position_encoder(positions)
        force_vectors = self.force_encoder(forces)
        return torch.stack((position_vectors, force_vectors), dim=-3)


class ScalarFiLM(nn.Module):
    """Produce [B,5,C] gains from pooled slow tokens and broadcast over XYZ."""

    def __init__(self, context_dim: int, channels: int):
        super().__init__()
        if context_dim <= 0 or channels <= 0:
            raise ValueError("context_dim and channels must be positive")
        self.context_dim = context_dim
        self.channels = channels
        self.gain_network = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, 5 * channels),
        )

    def gains(self, slow_tokens: torch.Tensor) -> torch.Tensor:
        if slow_tokens.ndim != 3 or slow_tokens.shape[-1] != self.context_dim or slow_tokens.shape[1] == 0:
            raise ValueError(f"Expected nonempty slow tokens [B,N,{self.context_dim}]")
        pooled = slow_tokens.mean(dim=1)
        return self.gain_network(pooled).reshape(slow_tokens.shape[0], 5, self.channels)

    def forward(self, field: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
        _validate_field(field, self.channels)
        if gains.shape != (field.shape[0], 5, self.channels):
            raise ValueError(f"Expected gains [B,5,{self.channels}] matching the field batch")
        return field * (1 + gains[:, None, :, None, :, None])


class FutureForcePredictor(nn.Module):
    """Predict [B,T,D,5,C,3] future-force carriers from current typed state.

    Only force vectors enter the vector projection. Position information
    contributes scalar gates derived from its squared vector norms, so Rp has
    no effect on the prediction and Rf rotates the prediction. No predicted
    future is recursively fed into this head. Its input is the same modulated
    field supplied to the action adapter; the caller controls EMA targets.
    """

    def __init__(self, channels: int, horizon: int = 8):
        super().__init__()
        if channels <= 0 or horizon <= 0:
            raise ValueError("channels and horizon must be positive")
        self.channels = channels
        self.horizon = horizon
        self.force_projection = VectorLinear(channels, horizon * channels)
        self.position_gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, horizon * channels),
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        _validate_field(field, self.channels)
        position_vectors = field[..., 0, :, :]
        force_vectors = field[..., 1, :, :]
        invariants = torch.log1p(position_vectors.square().sum(-1))
        gains = 1 + torch.tanh(self.position_gate(invariants))
        prediction = self.force_projection(force_vectors) * gains.unsqueeze(-1)
        batch, steps = field.shape[:2]
        prediction = prediction.reshape(batch, steps, 5, self.horizon, self.channels, 3)
        return prediction.permute(0, 1, 3, 2, 4, 5).contiguous()
