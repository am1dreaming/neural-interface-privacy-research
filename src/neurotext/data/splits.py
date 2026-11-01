"""Leakage-safe splits.

The most common failure in EEG-to-text work is evaluating on sentences or people
the model has already seen. Every split here is built from *groups* (subjects
and/or sentences), and `check_no_leakage` can verify any split after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class LeakageError(AssertionError):
    """Raised when train/val/test share a subject or a sentence."""


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    # which groups went where, e.g. {"train_subjects": [...], "test_sentences": [...]}
    groups: dict = field(default_factory=dict)


def _partition_groups(
    unique_groups: np.ndarray, val_frac: float, test_frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(unique_groups)
    if n < 3:
        raise ValueError(f"need at least 3 groups to split, got {n}")
    if not (0 <= val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError("fractions must satisfy 0 <= val, 0 < test, val + test < 1")
    perm = rng.permutation(unique_groups)
    n_test = max(1, int(round(test_frac * n)))
    n_val = max(1, int(round(val_frac * n))) if val_frac > 0 else 0
    if n_test + n_val >= n:
        raise ValueError("too few groups for the requested fractions")
    test = perm[:n_test]
    val = perm[n_test : n_test + n_val]
    train = perm[n_test + n_val :]
    return np.sort(train), np.sort(val), np.sort(test)


def _partition_sentences(sentence_ids, val_frac, test_frac, rng, block_size=None):
    """Random partition of sentences, or of contiguous blocks of `block_size` sentences.

    Blocks keep neighbours together. That matters when sentence ids follow the order of
    presentation (true for ZuCo): neighbouring sentences share a topic, and slow drifts in the
    recording make neighbouring recordings similar, so a random split lets a model profit from
    "which sentences were seen just before and after" instead of from the language itself.
    """
    ids = np.unique(sentence_ids)
    if block_size is None:
        return _partition_groups(ids, val_frac, test_frac, rng)
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    blocks = np.unique(ids // block_size)
    tr_b, va_b, te_b = _partition_groups(blocks, val_frac, test_frac, rng)
    expand = lambda b: ids[np.isin(ids // block_size, b)]
    return expand(tr_b), expand(va_b), expand(te_b)


def _indices_for(groups: np.ndarray, member_of: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(member_of, groups))


def subject_split(
    subject_ids: np.ndarray, val_frac: float = 0.15, test_frac: float = 0.25, seed: int = 0
) -> Split:
    """Hold out whole people. Use for 'does it work on a new person?'."""
    rng = np.random.default_rng(seed)
    tr, va, te = _partition_groups(np.unique(subject_ids), val_frac, test_frac, rng)
    return Split(
        _indices_for(tr, subject_ids),
        _indices_for(va, subject_ids),
        _indices_for(te, subject_ids),
        {"train_subjects": tr, "val_subjects": va, "test_subjects": te},
    )


def sentence_split(
    sentence_ids: np.ndarray, val_frac: float = 0.15, test_frac: float = 0.25, seed: int = 0,
    block_size: int | None = None,
) -> Split:
    """Hold out whole sentences (or contiguous blocks of them). Use for 'does it work on unseen text?'."""
    rng = np.random.default_rng(seed)
    tr, va, te = _partition_sentences(sentence_ids, val_frac, test_frac, rng, block_size)
    return Split(
        _indices_for(tr, sentence_ids),
        _indices_for(va, sentence_ids),
        _indices_for(te, sentence_ids),
        {"train_sentences": tr, "val_sentences": va, "test_sentences": te},
    )


def subject_and_sentence_split(
    subject_ids: np.ndarray,
    sentence_ids: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.25,
    seed: int = 0,
    block_size: int | None = None,
) -> Split:
    """Strictest split: test = unseen people reading unseen sentences.

    A sample is kept only if *both* its subject and its sentence belong to the same
    partition; mixed samples (e.g. a train subject reading a test sentence) are dropped,
    because they would leak one of the two factors.
    """
    rng = np.random.default_rng(seed)
    s_tr, s_va, s_te = _partition_groups(np.unique(subject_ids), val_frac, test_frac, rng)
    t_tr, t_va, t_te = _partition_sentences(sentence_ids, val_frac, test_frac, rng, block_size)

    def pick(subj, sent):
        return np.flatnonzero(np.isin(subject_ids, subj) & np.isin(sentence_ids, sent))

    return Split(
        pick(s_tr, t_tr),
        pick(s_va, t_va),
        pick(s_te, t_te),
        {
            "train_subjects": s_tr, "val_subjects": s_va, "test_subjects": s_te,
            "train_sentences": t_tr, "val_sentences": t_va, "test_sentences": t_te,
        },
    )


def check_no_leakage(
    split: Split,
    subject_ids: np.ndarray,
    sentence_ids: np.ndarray,
    by: tuple[str, ...] = ("subject", "sentence"),
) -> None:
    """Raise `LeakageError` if partitions overlap in samples, subjects, or sentences."""
    parts = {"train": split.train, "val": split.val, "test": split.test}
    names = list(parts)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if np.intersect1d(parts[a], parts[b]).size:
                raise LeakageError(f"{a} and {b} share samples")
            if "subject" in by:
                shared = np.intersect1d(subject_ids[parts[a]], subject_ids[parts[b]])
                if shared.size:
                    raise LeakageError(f"{a} and {b} share subjects: {shared[:5].tolist()}")
            if "sentence" in by:
                shared = np.intersect1d(sentence_ids[parts[a]], sentence_ids[parts[b]])
                if shared.size:
                    raise LeakageError(f"{a} and {b} share sentences: {shared[:5].tolist()}")

    if split.train.size == 0 or split.test.size == 0:
        raise LeakageError("train and test must both be non-empty")


def select_calibration(
    subject_ids: np.ndarray,
    sentence_ids: np.ndarray,
    subject: int,
    pool_sentences: np.ndarray,
    n_sentences: int,
    seed: int = 0,
) -> np.ndarray:
    """Pick calibration samples for a new person.

    Takes all trials of `n_sentences` sentences drawn from `pool_sentences` (which must
    be disjoint from the evaluation sentences). Selecting by sentence, not by trial,
    keeps repeated readings of one sentence from appearing in both calibration and eval.
    """
    if n_sentences <= 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    available = np.intersect1d(np.unique(sentence_ids[subject_ids == subject]), pool_sentences)
    if n_sentences > available.size:
        raise ValueError(f"asked for {n_sentences} calibration sentences, only {available.size} available")
    chosen = rng.choice(available, size=n_sentences, replace=False)
    return np.flatnonzero((subject_ids == subject) & np.isin(sentence_ids, chosen))
