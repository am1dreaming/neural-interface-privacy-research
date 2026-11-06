"""Open-set verification: "do these two recordings come from the same person?"

Closed-set identification asks an attacker who knows the candidates to name one of them. Verification
needs no list of candidates: score a PAIR of recordings and decide "same" or "different". It is the
stricter test for new people, because nothing about them is learnt by the attacker beyond what a model
fitted on OTHER people carries over.

Scores:
  cosine   cosine similarity after subtracting the mean of the reference set (no labels used);
  lda      cosine in the space of linear discriminant directions fitted on OTHER people (`fit_x`, `fit_y`).

Pairs are drawn from different groups (sentence blocks / runs): two recordings made next to each other
share slow drifts and would make verification look easier than it is. Metrics: EER (0.5 = chance, lower
is more leakage) and AUC (0.5 = chance, higher is more leakage). Both are computed with numpy only.
"""

from __future__ import annotations

import numpy as np

SCORERS = ("cosine", "lda")


def eer_auc(scores: np.ndarray, same: np.ndarray) -> dict[str, float]:
    """Equal error rate and ROC AUC of scores where HIGHER means 'same person'."""
    scores = np.asarray(scores, dtype=np.float64)
    same = np.asarray(same, dtype=bool)
    n_pos, n_neg = int(same.sum()), int((~same).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both genuine and impostor pairs")
    # AUC through ranks (ties get the average rank): P(score_genuine > score_impostor) + 0.5 P(tie)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    ends = np.r_[starts[1:], len(scores)]
    for a, b in zip(starts, ends):
        ranks[order[a:b]] = 0.5 * (a + 1 + b)
    auc = (ranks[same].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    # EER: sweep the threshold over the distinct score values
    thr = np.unique(scores)
    thr = np.r_[thr[0] - 1.0, thr, thr[-1] + 1.0]
    g, i = np.sort(scores[same]), np.sort(scores[~same])
    frr = np.searchsorted(g, thr, side="left") / n_pos            # genuine rejected: score < thr
    far = 1.0 - np.searchsorted(i, thr, side="left") / n_neg      # impostor accepted: score >= thr
    k = int(np.argmin(np.abs(frr - far)))
    return {"eer": float(0.5 * (frr[k] + far[k])), "auc": float(auc)}


def make_pairs(subject_ids: np.ndarray, groups: np.ndarray, n_pairs: int = 20000, seed: int = 0):
    """`n_pairs` genuine and `n_pairs` impostor index pairs (i, j), always from different groups.
    Returns (i, j, same). Fewer pairs come back if the data cannot supply that many."""
    subject_ids, groups = np.asarray(subject_ids), np.asarray(groups)
    rng = np.random.default_rng(seed)
    n = len(subject_ids)
    by_subject = {s: np.flatnonzero(subject_ids == s) for s in np.unique(subject_ids)}
    gi, gj, ii, ij = [], [], [], []
    for _ in range(n_pairs * 5):  # bounded retries
        if len(gi) >= n_pairs and len(ii) >= n_pairs:
            break
        a = int(rng.integers(n))
        if len(gi) < n_pairs:
            cand = by_subject[subject_ids[a]]
            cand = cand[groups[cand] != groups[a]]
            if cand.size:
                gi.append(a), gj.append(int(rng.choice(cand)))
        if len(ii) < n_pairs:
            b = int(rng.integers(n))
            if subject_ids[b] != subject_ids[a] and groups[b] != groups[a]:
                ii.append(a), ij.append(b)
    i = np.array(gi + ii, dtype=np.int64)
    j = np.array(gj + ij, dtype=np.int64)
    same = np.r_[np.ones(len(gi), bool), np.zeros(len(ii), bool)]
    return i, j, same


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(den, 1e-12)


def verification(
    x: np.ndarray,
    subject_ids: np.ndarray,
    groups: np.ndarray,
    fit_x: np.ndarray | None = None,
    fit_y: np.ndarray | None = None,
    n_pairs: int = 20000,
    seed: int = 0,
) -> dict:
    """Verification of the people in `x` from pairs of their recordings made in different groups.

    `fit_x`, `fit_y`: representations and person labels of people who are NOT in `x`. They give the
    reference mean for the cosine score and the discriminant directions for the LDA score. Without them
    only the cosine score (centred on the mean of `x`) is returned.
    """
    x = np.asarray(x, dtype=np.float64)
    i, j, same = make_pairs(subject_ids, groups, n_pairs, seed)
    if not same.any() or same.all():
        raise ValueError("could not form genuine and impostor pairs from different groups")
    centre = np.asarray(fit_x, dtype=np.float64).mean(axis=0) if fit_x is not None else x.mean(axis=0)
    out: dict = {"n_genuine": int(same.sum()), "n_impostor": int((~same).sum()),
                 "cosine": eer_auc(_cosine(x[i] - centre, x[j] - centre), same), "lda": None}
    if fit_x is not None and fit_y is not None and len(np.unique(fit_y)) >= 3:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="svd"))
        lda.fit(np.asarray(fit_x, dtype=np.float64), np.asarray(fit_y))
        z = lda.transform(x)
        out["lda"] = eer_auc(_cosine(z[i], z[j]), same)
    return out
