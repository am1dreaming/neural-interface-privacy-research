"""Aligning brain embeddings with text embeddings (CLIP-style contrastive learning)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TextProjector(nn.Module):
    """Linear map from frozen text embeddings into the shared space."""

    def __init__(self, text_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, out_dim, bias=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(t), dim=-1)


def info_nce(
    z_brain: torch.Tensor,
    z_text: torch.Tensor,
    sentence_ids: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Symmetric InfoNCE with in-batch negatives.

    Recordings of the same sentence in one batch (other trials or other people) are NOT
    negatives of each other: they are masked out, otherwise the loss punishes the model
    for being right.
    """
    logits = z_brain @ z_text.T / temperature
    same = sentence_ids[:, None] == sentence_ids[None, :]
    eye = torch.eye(len(sentence_ids), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(same & ~eye, float("-inf"))
    target = torch.arange(len(sentence_ids), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
