"""A small EEG encoder: temporal patches -> transformer -> L2-normalised embedding.

Deliberately tiny: the project cares about what fits on a wearable. Note that the
patch embedding is tied to a fixed channel count; channel-agnostic encoders (electrode
position embeddings) are a planned extension.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class EEGEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        d_model: int = 64,
        out_dim: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        patch: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if n_times % patch:
            raise ValueError(f"n_times ({n_times}) must be divisible by patch ({patch})")
        self.n_channels = n_channels
        self.out_dim = out_dim
        self.patch_embed = nn.Conv1d(n_channels, d_model, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, n_times // patch, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

    @staticmethod
    def prepare(x: torch.Tensor) -> torch.Tensor:
        """Per-trial, per-channel z-score over time (removes DC offset and gain)."""
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)

    def encode_prepared(self, x: torch.Tensor) -> torch.Tensor:
        h = self.patch_embed(x).transpose(1, 2) + self.pos
        h = self.norm(self.transformer(h)).mean(dim=1)
        return F.normalize(self.head(h), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, out_dim), unit norm."""
        return self.encode_prepared(self.prepare(x))
