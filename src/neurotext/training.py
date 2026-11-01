"""Training loops: base alignment on many people, adapter fitting on one new person."""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from .models.adapters import AdaptedEncoder, SubjectAdapter, freeze
from .models.alignment import TextProjector, info_nce
from .models.encoder import EEGEncoder


def _batches(n: int, batch_size: int, rng: np.random.Generator):
    perm = rng.permutation(n)
    for i in range(0, n, batch_size):
        chunk = perm[i : i + batch_size]
        if len(chunk) >= 2:  # contrastive loss needs at least two samples
            yield chunk


def fit_alignment(
    encoder: EEGEncoder,
    projector: TextProjector,
    x: np.ndarray,
    sentence_idx: np.ndarray,
    text_emb: np.ndarray,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    temperature: float = 0.1,
    seed: int = 0,
) -> list[float]:
    """Train encoder + projector contrastively. Returns the mean loss per epoch."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xt = torch.as_tensor(x, dtype=torch.float32)
    sent = torch.as_tensor(sentence_idx, dtype=torch.long)
    txt = torch.as_tensor(text_emb, dtype=torch.float32)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(projector.parameters()), lr=lr, weight_decay=1e-2)

    history = []
    encoder.train(), projector.train()
    for _ in range(epochs):
        losses = []
        for b in _batches(len(xt), batch_size, rng):
            loss = info_nce(encoder(xt[b]), projector(txt[sent[b]]), sent[b], temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
    encoder.eval(), projector.eval()
    return history


class _GradientReversal(torch.autograd.Function):
    """Identity on the forward pass; multiplies the gradient by -weight on the way back."""

    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.weight * grad, None


def fit_alignment_adversarial(
    encoder: EEGEncoder,
    projector: TextProjector,
    x: np.ndarray,
    sentence_idx: np.ndarray,
    subject_idx: np.ndarray,
    text_emb: np.ndarray,
    *,
    adv_weight: float,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    temperature: float = 0.1,
    seed: int = 0,
    adv_hidden: int = 64,
) -> tuple[list[float], list[float]]:
    """Train the encoder to match text AND to hide who the person is.

    A small adversary tries to name the person from the embedding; the encoder receives the
    reversed gradient of the adversary's loss (scaled by `adv_weight`), so it is pushed to make that
    impossible while still minimising the contrastive loss. `subject_idx` must be 0..K-1 for the K
    training people. Returns (contrastive loss per epoch, adversary loss per epoch). With
    `adv_weight=0` the encoder behaves as in `fit_alignment`, apart from the extra adversary."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n_people = int(np.max(subject_idx)) + 1
    adversary = nn.Sequential(nn.Linear(encoder.out_dim, adv_hidden), nn.ReLU(), nn.Linear(adv_hidden, n_people))
    xt = torch.as_tensor(x, dtype=torch.float32)
    sent = torch.as_tensor(sentence_idx, dtype=torch.long)
    who = torch.as_tensor(subject_idx, dtype=torch.long)
    txt = torch.as_tensor(text_emb, dtype=torch.float32)
    params = list(encoder.parameters()) + list(projector.parameters()) + list(adversary.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)

    task_hist, adv_hist = [], []
    encoder.train(), projector.train(), adversary.train()
    for _ in range(epochs):
        task, adv = [], []
        for b in _batches(len(xt), batch_size, rng):
            z = encoder(xt[b])
            task_loss = info_nce(z, projector(txt[sent[b]]), sent[b], temperature)
            adv_loss = torch.nn.functional.cross_entropy(adversary(_GradientReversal.apply(z, adv_weight)), who[b])
            opt.zero_grad()
            (task_loss + adv_loss).backward()
            opt.step()
            task.append(task_loss.item())
            adv.append(adv_loss.item())
        task_hist.append(float(np.mean(task)))
        adv_hist.append(float(np.mean(adv)))
    encoder.eval(), projector.eval()
    return task_hist, adv_hist


def fit_adapter(
    encoder: EEGEncoder,
    projector: TextProjector,
    x: np.ndarray,
    sentence_idx: np.ndarray,
    text_emb: np.ndarray,
    *,
    rank: int = 4,
    epochs: int = 40,
    lr: float = 1e-2,
    batch_size: int = 32,
    temperature: float = 0.1,
    seed: int = 0,
    components: tuple[str, ...] = ("channel", "lowrank"),
    identity_reg: float = 0.0,
    negative_pool: np.ndarray | None = None,
) -> AdaptedEncoder:
    """Fit a SubjectAdapter on a small calibration set. Encoder and projector stay frozen
    (the originals are untouched: frozen copies are used). With no data, the adapter is
    the identity and the result equals the base model.

    components:    which adapter parts are trained: "channel" (per-channel scale/bias) and/or
                   "lowrank" (residual on the embedding).
    identity_reg:  L2 penalty on the distance from the identity adapter (0 = off).
    negative_pool: sentence ids whose texts all serve as negatives (the texts of known
                   sentences are free). None = only the other sentences in the mini-batch,
                   which with 2-8 calibration sentences is a very easy contrastive task.
                   The pool must not contain evaluation sentences.
    """
    unknown = set(components) - {"channel", "lowrank", "mixing"}
    if unknown:
        raise ValueError(f"unknown adapter components: {sorted(unknown)}")
    enc = freeze(copy.deepcopy(encoder))
    proj = freeze(copy.deepcopy(projector))
    torch.manual_seed(seed)  # the adapter's random init must depend on the seed too
    adapter = SubjectAdapter(enc.n_channels, enc.out_dim, rank=rank)
    adapted = AdaptedEncoder(enc, adapter)
    if len(x) < 2 or not components:
        return adapted.eval()

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    xt = torch.as_tensor(x, dtype=torch.float32)
    sent = torch.as_tensor(sentence_idx, dtype=torch.long)
    txt = torch.as_tensor(text_emb, dtype=torch.float32)

    params = []
    if "channel" in components:
        params += [adapter.scale, adapter.bias]
    if "lowrank" in components:
        params += list(adapter.down.parameters()) + list(adapter.up.parameters())
    if "mixing" in components:
        params += list(adapter.mix_down.parameters()) + list(adapter.mix_up.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    pool_z = pool_pos = None
    if negative_pool is not None:
        pool = np.asarray(negative_pool)
        missing = set(np.unique(sentence_idx)) - set(pool.tolist())
        if missing:
            raise ValueError(f"calibration sentences missing from negative_pool: {sorted(missing)[:5]}")
        with torch.no_grad():
            pool_z = proj(txt[torch.as_tensor(pool, dtype=torch.long)])
        position = {int(s): i for i, s in enumerate(pool)}
        pool_pos = torch.as_tensor([position[int(s)] for s in sentence_idx], dtype=torch.long)

    for _ in range(epochs):
        for b in _batches(len(xt), batch_size, rng):
            z = adapted(xt[b])
            if pool_z is None:
                with torch.no_grad():
                    zt = proj(txt[sent[b]])
                loss = info_nce(z, zt, sent[b], temperature)
            else:
                loss = torch.nn.functional.cross_entropy(z @ pool_z.T / temperature, pool_pos[b])
            if identity_reg > 0:
                loss = loss + identity_reg * (
                    ((adapter.scale - 1) ** 2).sum() + (adapter.bias ** 2).sum()
                    + (adapter.up.weight ** 2).sum() + (adapter.mix_up.weight ** 2).sum()
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
    return adapted.eval()


@torch.no_grad()
def embed(model: nn.Module, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Embed recordings in eval mode, as a numpy array (N, d)."""
    model.eval()
    xt = torch.as_tensor(x, dtype=torch.float32)
    out = [model(xt[i : i + batch_size]) for i in range(0, len(xt), batch_size)]
    return torch.cat(out).numpy()


@torch.no_grad()
def project_text(projector: TextProjector, text_emb: np.ndarray) -> np.ndarray:
    projector.eval()
    return projector(torch.as_tensor(text_emb, dtype=torch.float32)).numpy()
