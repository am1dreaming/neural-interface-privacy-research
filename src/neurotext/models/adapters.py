"""Lightweight per-person adaptation on top of a frozen base model.

The adapter has two tiny parts: a per-channel scale/bias applied to the (already
normalised) recording, and a low-rank residual on the embedding. Both start as the
identity, so zero calibration data means exactly the base model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import EEGEncoder


class SubjectAdapter(nn.Module):
    """Three tiny identity-initialised parts:
    - scale/bias:  per-channel gain and offset on the recording;
    - mix_*:       low-rank residual that mixes channels (x + U V^T x), the kind of change
                   electrode placement and head geometry cause;
    - down/up:     low-rank residual on the embedding.
    """

    def __init__(self, n_channels: int, emb_dim: int, rank: int = 4, mix_rank: int = 4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, n_channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, n_channels, 1))
        self.down = nn.Linear(emb_dim, rank, bias=False)
        self.up = nn.Linear(rank, emb_dim, bias=False)
        nn.init.zeros_(self.up.weight)  # residual starts at zero -> identity
        # created after the parts above so existing random draws (and results) are unchanged
        self.mix_down = nn.Linear(n_channels, mix_rank, bias=False)
        self.mix_up = nn.Linear(mix_rank, n_channels, bias=False)
        nn.init.zeros_(self.mix_up.weight)

    def transform_input(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale + self.bias
        mixed = self.mix_up(self.mix_down(x.transpose(1, 2))).transpose(1, 2)  # (B, C, T)
        return x + mixed

    def transform_embedding(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(z + self.up(self.down(z)), dim=-1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AdaptedEncoder(nn.Module):
    def __init__(self, encoder: EEGEncoder, adapter: SubjectAdapter):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.adapter.transform_input(self.encoder.prepare(x))
        return self.adapter.transform_embedding(self.encoder.encode_prepared(h))


def freeze(module: nn.Module) -> nn.Module:
    """Freeze all parameters and switch to eval mode (no dropout during adaptation)."""
    for p in module.parameters():
        p.requires_grad_(False)
    return module.eval()
