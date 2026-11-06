"""Interchangeable encoders for the robustness check.

The audit should not depend on one architecture, so four encoders share one interface:

    encoder(x)  with x: (B, C, T)  ->  (B, out_dim), unit norm
    encoder.prepare(x)              per-trial, per-channel z-score (same as `EEGEncoder`)
    encoder.n_channels, encoder.out_dim

  transformer   the project's small patch transformer (`EEGEncoder`);
  linear        one linear map of the whole (z-scored) recording: the weakest sensible baseline;
  cnn           three temporal convolutions over all channels, global average pooling;
  eegnet        EEGNet-8,2 (Lawhern et al. 2018): temporal conv, depthwise spatial conv, separable conv.

`build_encoder` passes each encoder only the options it understands, so one config can describe all of them.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import EEGEncoder


class _Prepared(nn.Module):
    """Shared plumbing: z-score the recording, run `encode_prepared`, L2-normalise."""

    n_channels: int
    out_dim: int

    prepare = staticmethod(EEGEncoder.prepare)

    def encode_prepared(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_prepared(self.prepare(x))


class LinearEncoder(_Prepared):
    def __init__(self, n_channels: int, n_times: int, out_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        self.n_channels, self.out_dim = n_channels, out_dim
        self.drop = nn.Dropout(dropout)
        self.lin = nn.Linear(n_channels * n_times, out_dim)

    def encode_prepared(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.lin(self.drop(x.flatten(1))), dim=-1)


class SmallCNN(_Prepared):
    def __init__(self, n_channels: int, n_times: int, out_dim: int = 32, width: int = 32,
                 kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        self.n_channels, self.out_dim = n_channels, out_dim
        pad = kernel // 2

        def block(cin, cout, pool):
            layers = [nn.Conv1d(cin, cout, kernel, padding=pad), nn.BatchNorm1d(cout), nn.GELU()]
            return layers + ([nn.MaxPool1d(pool)] if pool else [])

        self.body = nn.Sequential(
            *block(n_channels, width, 4), *block(width, 2 * width, 4), *block(2 * width, 2 * width, 0),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(2 * width, out_dim)

    def encode_prepared(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.body(x)), dim=-1)


class EEGNet(_Prepared):
    """EEGNet-8,2 with an embedding head instead of the classifier. Kernel sizes are odd so that
    'same' padding keeps the length."""

    def __init__(self, n_channels: int, n_times: int, out_dim: int = 32, f1: int = 8, depth: int = 2,
                 f2: int = 16, kernel: int = 33, sep_kernel: int = 15, dropout: float = 0.25):
        super().__init__()
        self.n_channels, self.out_dim = n_channels, out_dim
        d = f1 * depth
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel), padding=(0, kernel // 2), bias=False), nn.BatchNorm2d(f1),
            nn.Conv2d(f1, d, (n_channels, 1), groups=f1, bias=False), nn.BatchNorm2d(d), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(d, d, (1, sep_kernel), padding=(0, sep_kernel // 2), groups=d, bias=False),
            nn.Conv2d(d, f2, 1, bias=False), nn.BatchNorm2d(f2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout), nn.Flatten(),
        )
        was_training = self.training
        self.eval()  # BatchNorm cannot take a single-sample dummy batch in train mode
        try:
            with torch.no_grad():
                flat = self.features(torch.zeros(1, 1, n_channels, n_times)).shape[1]
        except RuntimeError as err:
            raise ValueError(f"n_times={n_times} is too short for EEGNet (needs at least 32 samples)") from err
        finally:
            self.train(was_training)
        if flat == 0:
            raise ValueError(f"n_times={n_times} is too short for EEGNet (needs at least 32 samples)")
        self.head = nn.Linear(flat, out_dim)

    def encode_prepared(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.features(x.unsqueeze(1))), dim=-1)


ENCODERS = {"transformer": EEGEncoder, "linear": LinearEncoder, "cnn": SmallCNN, "eegnet": EEGNet}


def build_encoder(name: str, n_channels: int, n_times: int, **options) -> nn.Module:
    """Build `name`, keeping only the options its constructor accepts (so a transformer config with
    `d_model`, `n_heads`, `patch` can be reused for the others, which ignore them)."""
    try:
        cls = ENCODERS[name]
    except KeyError:
        raise ValueError(f"unknown encoder {name!r}; choose from {sorted(ENCODERS)}") from None
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self", "n_channels", "n_times"}
    return cls(n_channels, n_times, **{k: v for k, v in options.items() if k in accepted})


def model_options(model_cfg: dict) -> tuple[str, dict]:
    """Split a `model:` config block into (encoder name, constructor options). `name` defaults to
    'transformer', so every existing config keeps working."""
    opts = dict(model_cfg)
    return opts.pop("name", "transformer"), opts
