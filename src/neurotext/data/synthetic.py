"""Synthetic EEG-like data for smoke tests.

This is NOT a model of real EEG. It exists so the whole pipeline (splits, training,
baselines, privacy probe, profiling) can run end to end without any dataset. Never
report numbers obtained here as research results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticData:
    x: np.ndarray            # (N, C, T) float32
    subject_ids: np.ndarray  # (N,)
    sentence_ids: np.ndarray # (N,)
    text_emb: np.ndarray     # (S, D) float32, row i = sentence i, L2-normalised


def make_synthetic(
    n_subjects: int = 12,
    n_sentences: int = 64,
    trials_per_sentence: int = 2,
    n_channels: int = 16,
    n_times: int = 128,
    text_dim: int = 16,
    n_sources: int = 8,
    snr: float = 1.0,
    subject_variability: float = 0.3,
    seed: int = 0,
) -> SyntheticData:
    """Each sentence has a latent text embedding. A fixed linear map turns it into source
    amplitudes, sources have fixed temporal shapes, and each subject mixes sources into
    channels with a subject-specific perturbation of a shared mixing matrix."""
    rng = np.random.default_rng(seed)

    text_emb = rng.standard_normal((n_sentences, text_dim))
    text_emb /= np.linalg.norm(text_emb, axis=1, keepdims=True)

    to_sources = rng.standard_normal((text_dim, n_sources))
    t = np.linspace(0, 1, n_times)
    freqs = rng.uniform(2, 12, size=n_sources)
    phases = rng.uniform(0, 2 * np.pi, size=n_sources)
    basis = np.sin(2 * np.pi * freqs[:, None] * t[None, :] + phases[:, None])  # (K, T)

    base_mix = rng.standard_normal((n_channels, n_sources))

    xs, subj, sent = [], [], []
    for s in range(n_subjects):
        mix = base_mix + subject_variability * rng.standard_normal(base_mix.shape)
        for k in range(n_sentences):
            amp = text_emb[k] @ to_sources                     # (K,)
            sources = amp[:, None] * basis                     # (K, T)
            clean = mix @ sources                              # (C, T)
            clean /= clean.std() + 1e-8
            for _ in range(trials_per_sentence):
                noise = rng.standard_normal(clean.shape)
                xs.append(snr * clean + noise)
                subj.append(s)
                sent.append(k)

    return SyntheticData(
        x=np.asarray(xs, dtype=np.float32),
        subject_ids=np.asarray(subj),
        sentence_ids=np.asarray(sent),
        text_emb=text_emb.astype(np.float32),
    )
