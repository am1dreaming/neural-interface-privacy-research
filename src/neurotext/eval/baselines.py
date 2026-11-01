"""Baselines every experiment must report next to its own number.

The signal's contribution is (model - baselines), not the raw metric. Known pitfalls in
EEG-to-text work: models that score the same on noise input, and language-model priors
doing the work without any brain signal.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .metrics import chance_topk, topk_retrieval


def noise_like(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise with the same per-channel mean and std as `x`: (N, C, T) recordings or
    (N, F) feature tables (axis 1 is the channel/feature axis)."""
    axes = tuple(a for a in range(x.ndim) if a != 1)
    mean = x.mean(axis=axes, keepdims=True)
    std = x.std(axis=axes, keepdims=True)
    return (mean + std * rng.standard_normal(x.shape)).astype(x.dtype)


def run_baselines(
    embed_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    true_idx: np.ndarray,
    candidates: np.ndarray,
    *,
    prior_logprob: np.ndarray | None = None,
    ks: tuple[int, ...] = (1, 5),
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Score the model and its baselines on the same recordings.

    embed_fn:   recordings (N, C, T) -> embeddings (N, d), in the candidates' space.
    true_idx:   (N,) index of the correct candidate for each recording.
    candidates: (S, d) candidate sentence embeddings.
    prior_logprob: optional (S,) text-only prior (e.g. language-model log-probability of
        each candidate). It is scored with no brain signal at all.
    """
    rng = np.random.default_rng(seed)
    n = len(true_idx)

    sim = embed_fn(x) @ candidates.T
    out = {"model": topk_retrieval(sim, true_idx, ks)}

    sim_noise = embed_fn(noise_like(x, rng)) @ candidates.T
    out["noise_input"] = topk_retrieval(sim_noise, true_idx, ks)

    out["shuffled_labels"] = topk_retrieval(sim, rng.permutation(true_idx), ks)
    out["chance"] = chance_topk(candidates.shape[0], ks)

    if prior_logprob is not None:
        out["prior_only"] = topk_retrieval(np.tile(prior_logprob, (n, 1)), true_idx, ks)
    return out


def signal_gain(results: dict[str, dict[str, float]], metric: str = "top1") -> float:
    """Model score minus the strongest baseline for `metric`. Near zero or negative means
    the model has not shown that it uses the brain signal."""
    baselines = [v[metric] for k, v in results.items() if k != "model" and metric in v]
    return results["model"][metric] - max(baselines)
