"""The one interface every dataset loader returns.

Everything downstream (splits, training, baselines, probes) only needs these fields, so
adding a dataset means writing one reader module that produces an `EEGData`; nothing else
changes. `SyntheticData` has the same four core fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EEGData:
    x: np.ndarray             # (N, C, T) float32, one fixed-length window per recording
    subject_ids: np.ndarray   # (N,) int, one id per person
    sentence_ids: np.ndarray  # (N,) int, index into `text_emb` / `sentences`
    text_emb: np.ndarray      # (S, D) float32, L2-normalised, row i = sentence i
    sentences: list[str] = field(default_factory=list)       # (S,) text of each sentence id
    subject_codes: list[str] = field(default_factory=list)   # original name of each subject id
    sfreq: float | None = None                                # sampling rate of `x`, Hz
    meta: dict = field(default_factory=dict)                  # dropped-recording report etc.

    def __post_init__(self):
        n = len(self.x)
        if not (len(self.subject_ids) == len(self.sentence_ids) == n):
            raise ValueError("x, subject_ids and sentence_ids must have the same length")
        if self.sentence_ids.size and self.sentence_ids.max() >= len(self.text_emb):
            raise ValueError("a sentence id points past the end of text_emb")
