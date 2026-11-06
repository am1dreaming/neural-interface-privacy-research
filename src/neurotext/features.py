"""Interpretable features, to ask WHERE identity leakage comes from.

A leaked embedding says "a person can be recognised". These features say what the recognition rests on:

  rms          log amplitude of every channel (electrode impedance, skull thickness, gain);
  delta ... gamma   log power of every channel in the classical bands (the shape of the spectrum);
  alpha_peak   individual alpha peak: frequency and log power of the strongest peak in 7-14 Hz, per channel.

Feature GROUPS can be kept alone or removed one at a time (`FeatureTable.columns`). Two controls look for
recording artefacts rather than physiology:

  amplitude_normalise   divide every channel of every recording by its own standard deviation (a
                        per-RECORDING step, so it needs no knowledge of who the person is);
  trim_edges            cut a fraction of the window at both ends, where filter transients and slow
                        drift sit.

Deliberately NOT offered: normalising by a person's own statistics. Inside cross-validation it uses the
label the attacker is looking for and gives accuracy far below chance (see `leakage.py`).

Numpy only (the Welch spectrum is written out), so it does not depend on the scipy version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

BANDS_HZ = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0)}
ALPHA_SEARCH_HZ = (7.0, 14.0)
GROUPS = ("rms", *BANDS_HZ, "alpha_peak")

_REGIONS = {"FP": "frontal", "AF": "frontal", "F": "frontal", "FC": "central", "C": "central",
            "FT": "temporal", "T": "temporal", "TP": "temporal", "CP": "parietal", "P": "parietal",
            "PO": "occipital", "O": "occipital", "I": "occipital"}


def region_of(channel_name: str) -> str:
    """Coarse scalp region of a 10-10 channel name such as 'Fc5.', 'Cz..', 'Po7'."""
    letters = re.match(r"[A-Za-z]+", channel_name.strip().strip("."))
    if not letters:
        return "other"
    code = letters.group().upper()
    if len(code) > 1 and code.endswith("Z"):
        code = code[:-1]
    return _REGIONS.get(code, "other")


def welch_psd(x: np.ndarray, sfreq: float, nperseg: int | None = None, batch: int = 256):
    """Welch power spectral density along the last axis of (N, C, T): Hann window, 50% overlap,
    per-segment mean removed. Returns (frequencies, psd (N, C, F) float32)."""
    n, c, t = x.shape
    nperseg = min(t, nperseg or int(4 * sfreq))
    step = max(1, nperseg // 2)
    starts = list(range(0, t - nperseg + 1, step)) or [0]
    win = np.hanning(nperseg).astype(np.float32)
    scale = 1.0 / (sfreq * float(np.sum(win ** 2)))
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sfreq)
    weight = np.full(len(freqs), 2.0, dtype=np.float32)  # one-sided: double all but DC and Nyquist
    weight[0] = 1.0
    if nperseg % 2 == 0:
        weight[-1] = 1.0
    out = np.empty((n, c, len(freqs)), dtype=np.float32)
    for b in range(0, n, batch):
        chunk = x[b : b + batch].astype(np.float32, copy=False)
        acc = np.zeros((len(chunk), c, len(freqs)), dtype=np.float32)
        for s in starts:
            seg = chunk[:, :, s : s + nperseg]
            seg = (seg - seg.mean(axis=-1, keepdims=True)) * win
            acc += np.abs(np.fft.rfft(seg, axis=-1)) ** 2
        out[b : b + batch] = acc / len(starts) * scale * weight
    return freqs, out


@dataclass(frozen=True)
class FeatureTable:
    x: np.ndarray                      # (N, d) float32
    groups: dict[str, np.ndarray]      # group name -> column indices
    channel: np.ndarray                # (d,) channel index of every column

    def columns(self, only: tuple[str, ...] | None = None, drop: tuple[str, ...] = ()) -> np.ndarray:
        """Column indices of the chosen groups: `only` (default all) minus `drop`."""
        names = [g for g in (only if only is not None else self.groups) if g not in drop]
        unknown = set(names) - set(self.groups)
        if unknown:
            raise ValueError(f"unknown feature groups {sorted(unknown)}; available {sorted(self.groups)}")
        return np.sort(np.concatenate([self.groups[g] for g in names])) if names else np.array([], dtype=int)

    def of_channels(self, channels) -> np.ndarray:
        return np.flatnonzero(np.isin(self.channel, np.asarray(channels)))


def feature_table(x: np.ndarray, sfreq: float, batch: int = 256) -> FeatureTable:
    """All feature groups for recordings x of shape (N, C, T). Bands that the sampling rate cannot
    resolve (upper edge capped at 90% of the Nyquist frequency, minimum width 2 Hz) are left out."""
    n, c, _ = x.shape
    freqs, psd = welch_psd(x, sfreq, batch=batch)
    df = float(freqs[1] - freqs[0])
    blocks: list[np.ndarray] = []
    groups: dict[str, list[int]] = {}
    channel: list[int] = []

    def add(name: str, block: np.ndarray) -> None:
        start = sum(b.shape[1] for b in blocks)
        blocks.append(block.astype(np.float32))
        groups.setdefault(name, []).extend(range(start, start + block.shape[1]))
        channel.extend(range(block.shape[1]))

    rms = np.empty((n, c), dtype=np.float32)
    for b in range(0, n, batch):  # in chunks: a full float32 copy of a big dataset is over a gigabyte
        chunk = x[b : b + batch].astype(np.float32)
        chunk -= chunk.mean(axis=-1, keepdims=True)
        rms[b : b + batch] = np.sqrt(np.mean(chunk ** 2, axis=-1))
    add("rms", np.log(rms + 1e-12))
    nyquist = sfreq / 2.0
    for name, (lo, hi) in BANDS_HZ.items():
        hi = min(hi, 0.9 * nyquist)
        if hi - lo < 2.0:
            continue
        sel = (freqs >= lo) & (freqs < hi)
        if sel.any():
            add(name, np.log(psd[:, :, sel].sum(axis=-1) * df + 1e-12))
    sel = (freqs >= ALPHA_SEARCH_HZ[0]) & (freqs <= ALPHA_SEARCH_HZ[1])
    if sel.any():
        sub = psd[:, :, sel]
        add("alpha_peak", freqs[sel][np.argmax(sub, axis=-1)])
        add("alpha_peak", np.log(sub.max(axis=-1) + 1e-12))
    return FeatureTable(np.concatenate(blocks, axis=1),
                        {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()},
                        np.asarray(channel, dtype=np.int64))


def amplitude_normalise(x: np.ndarray) -> np.ndarray:
    """Per recording and per channel: remove the mean, divide by the standard deviation. Removes the
    amplitude of every recording but keeps the shape of its spectrum."""
    out = np.empty(x.shape, dtype=np.float32)
    for b in range(0, len(x), 512):  # in chunks, to avoid several full-size temporaries
        xc = x[b : b + 512].astype(np.float32) - x[b : b + 512].mean(axis=-1, keepdims=True, dtype=np.float32)
        out[b : b + 512] = xc / (xc.std(axis=-1, keepdims=True) + 1e-8)
    return out


def trim_edges(x: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    """Cut `fraction` of the window from both ends."""
    if not 0 <= fraction < 0.5:
        raise ValueError("fraction must be in [0, 0.5)")
    k = int(round(fraction * x.shape[-1]))
    return x[..., k : x.shape[-1] - k] if k else x
