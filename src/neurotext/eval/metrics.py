"""Retrieval metrics: given a brain recording, pick the right sentence out of N candidates."""

from __future__ import annotations

import numpy as np


def topk_retrieval(sim: np.ndarray, true_idx: np.ndarray, ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    """`sim` is (N, S): similarity of each recording to each candidate sentence.
    `true_idx` is (N,): column index of the correct candidate. Ties count against the model."""
    sim = np.asarray(sim)
    true_idx = np.asarray(true_idx)
    if sim.ndim != 2 or sim.shape[0] != true_idx.shape[0]:
        raise ValueError("sim must be (N, S) and true_idx (N,)")
    true_score = sim[np.arange(len(true_idx)), true_idx][:, None]
    rank = (sim >= true_score).sum(axis=1)  # 1 = best; ties push the rank down
    return {f"top{k}": float((rank <= k).mean()) for k in ks if k <= sim.shape[1]}


def chance_topk(n_candidates: int, ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    """Expected top-k accuracy of random guessing."""
    return {f"top{k}": min(1.0, k / n_candidates) for k in ks if k <= n_candidates}
