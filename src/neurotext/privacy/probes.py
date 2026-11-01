"""Privacy probes: how much does an embedding reveal beyond what it is meant to?"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROBES = ("linear", "mlp")


def _make_probe(kind: str, seed: int, standardize: bool):
    if kind == "linear":
        clf = LogisticRegression(max_iter=1000)
    elif kind == "mlp":
        # a stronger, non-linear attacker: one hidden layer, weight decay, fixed seed
        clf = MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-2, max_iter=300, random_state=seed)
    else:
        raise ValueError(f"unknown probe {kind!r}; choose from {PROBES}")
    return make_pipeline(StandardScaler(), clf) if standardize else clf


def identity_leakage(
    embeddings: np.ndarray,
    subject_ids: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 4,
    seed: int = 0,
    probe: str = "linear",
    standardize: bool = False,
) -> dict[str, float]:
    """Train a probe to recover *who* produced each embedding.

    Returns mean probe accuracy over cross-validation folds, its std across folds, the
    chance level (1 / n_subjects), and the excess over chance. A linear probe is a lower
    bound on leakage: a stronger attacker (`probe="mlp"`) can only do better. Embeddings are
    meant to carry the text, so lower excess = better privacy.

    IMPORTANT: pass `groups` (e.g. sentence ids) whenever several recordings share the
    same text, or neighbours in time are correlated. If they are split at random, recordings
    of one sentence sit in both train and test, the probe learns "sentence -> majority label",
    and accuracy falls systematically *below* chance. With `groups`, whole groups stay on one side.

    `standardize` z-scores every feature on the training folds only. Turn it on for features
    on very different scales (e.g. raw band powers); it is off by default so earlier results
    stay reproducible.
    """
    embeddings = np.asarray(embeddings)
    subject_ids = np.asarray(subject_ids)
    subjects, counts = np.unique(subject_ids, return_counts=True)
    if len(subjects) < 2:
        raise ValueError("need at least two subjects")
    if counts.min() < n_splits:
        raise ValueError(f"every subject needs at least n_splits={n_splits} samples")

    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = splitter.split(embeddings, subject_ids)
    else:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = splitter.split(embeddings, subject_ids, groups=np.asarray(groups))

    scores = []
    for train, test in folds:
        if len(np.unique(subject_ids[train])) < len(subjects):
            continue  # a subject vanished from training: the fold cannot be scored fairly
        model = _make_probe(probe, seed, standardize)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(embeddings[train], subject_ids[train])
        scores.append(model.score(embeddings[test], subject_ids[test]))
    if not scores:
        raise ValueError("no usable folds; use more samples or fewer splits")

    acc = float(np.mean(scores))
    chance = 1.0 / len(subjects)
    return {
        "accuracy": acc,
        "accuracy_std_over_folds": float(np.std(scores)),
        "chance": chance,
        "excess_over_chance": acc - chance,
        "n_folds_used": len(scores),
    }
