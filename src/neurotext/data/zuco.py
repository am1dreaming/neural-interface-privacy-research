"""ZuCo 2.0, task 1 (normal reading): reader that returns the common `EEGData`.

Files: `task1 - NR/Matlab files/results<CODE>_NR.mat`, one per person, MATLAB v7.3 (HDF5).
Each holds `sentenceData`, a struct array over the 349 sentences; in HDF5 every field is an
(349, 1) array of object references. Fields used here:

    content     sentence text (uint16 character codes)
    rawData     EEG of the whole sentence, (samples, 105 channels); a scalar when missing
    wordbounds  (4, n_words) on-screen word boxes, used only to count words

Facts taken from the dataset paper (Hollenstein et al., LREC 2020), not from the files:
EEG sampling rate 500 Hz; 105 scalp channels kept after preprocessing; sentence order
identical for all participants (checked here: loading fails if the texts differ).
Not verified: what filtering/referencing `rawData` already went through.

Quality is person-dependent. Across the 16 people on disk, seconds of EEG per word form
two groups: a separate cluster below 0.03 s/word (55 recordings, 50 of them from one person,
e.g. 29 samples for 23 words), a nearly empty stretch from 0.03 to 0.06, and a dense normal
range from 0.06 up (fast readers such as YHS, whose minimum is 0.082, are inside it).
The default threshold of 0.05 s/word sits in that gap. Separately, between
0 and 102 sentences per person have no EEG at all (rawData is a scalar). Recordings are
filtered and what was dropped is reported in `EEGData.meta["dropped"]`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.signal import decimate

from .base import EEGData
from .text_features import tfidf_svd_embeddings

SFREQ = 500.0
N_CHANNELS = 105
_NAME = re.compile(r"^results(?P<code>[A-Za-z0-9]+)_NR\.mat$")


@dataclass
class SubjectRecordings:
    code: str
    sentences: list[str]       # all 349 texts, in file order
    x: np.ndarray              # (K, 105, n_times) float32, only the kept sentences
    sentence_idx: np.ndarray   # (K,) index into `sentences`
    dropped: dict              # reason -> count


def _h5py():
    try:
        import h5py
    except ImportError as e:  # pragma: no cover
        raise ImportError("reading ZuCo needs h5py: pip install h5py") from e
    return h5py


def _decode(f, ref) -> str:
    return "".join(chr(int(c)) for c in np.asarray(f[ref]).flatten())


def screen_recording(f, sd, i: int, min_seconds_per_word: float, min_samples: int = 0) -> str | None:
    """Why sentence `i` is unusable ("no_eeg" or "too_short"), or None if it passes. Shared by
    every ZuCo reader so that all of them keep exactly the same sentences."""
    h5py = _h5py()
    ds = f[sd["rawData"][i, 0]]
    if not isinstance(ds, h5py.Dataset) or ds.ndim != 2 or ds.shape[1] != N_CHANNELS:
        return "no_eeg"
    wb = f[sd["wordbounds"][i, 0]]
    n_words = wb.shape[1] if isinstance(wb, h5py.Dataset) and wb.ndim == 2 else 1
    if ds.shape[0] / SFREQ / max(n_words, 1) < min_seconds_per_word or ds.shape[0] < min_samples:
        return "too_short"
    return None


def _prepare_window(a: np.ndarray, factor: int, n_times: int) -> np.ndarray:
    """(105, T) float64 -> (105, n_times) float32: anti-aliased downsampling, z-score each
    channel over the real samples, then crop (keep the start) or zero-pad the end."""
    if factor > 1:
        a = decimate(a, factor, axis=1, zero_phase=True)
    std = a.std(axis=1, keepdims=True)
    a = (a - a.mean(axis=1, keepdims=True)) / np.where(std > 0, std, 1.0)
    out = np.zeros((a.shape[0], n_times), dtype=np.float32)
    k = min(a.shape[1], n_times)
    out[:, :k] = a[:, :k]
    return out


def read_subject_file(
    path: str | Path,
    *,
    decimate_factor: int = 5,
    n_times: int = 640,
    min_seconds_per_word: float = 0.05,
    cache_dir: str | Path | None = None,
) -> SubjectRecordings:
    """Read one person's file. Sentences are dropped (and counted) when they have

    - no_eeg:      rawData is not a (samples, 105) array;
    - too_short:   fewer than `min_seconds_per_word` seconds of EEG per word at 500 Hz, or
                   too few samples to downsample;
    - non_finite:  any NaN/inf.
    """
    path = Path(path)
    m = _NAME.match(path.name)
    code = m.group("code") if m else path.stem
    cache = None
    if cache_dir is not None:
        key = f"{code}_d{decimate_factor}_T{n_times}_m{min_seconds_per_word}_{path.stat().st_size}.npz"
        cache = Path(cache_dir) / key
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            return SubjectRecordings(code, [str(s) for s in z["sentences"]], z["x"], z["sentence_idx"],
                                     json.loads(str(z["dropped"])))

    h5py = _h5py()
    dropped = {"no_eeg": 0, "too_short": 0, "non_finite": 0}
    xs, idx = [], []
    with h5py.File(path, "r") as f:
        sd = f["sentenceData"]
        n = sd["content"].shape[0]
        sentences = [_decode(f, sd["content"][i, 0]) for i in range(n)]
        for i in range(n):
            reason = screen_recording(f, sd, i, min_seconds_per_word, min_samples=3 * decimate_factor * 10)
            if reason:
                dropped[reason] += 1
                continue
            ds = f[sd["rawData"][i, 0]]
            a = np.asarray(ds[()], dtype=np.float64).T  # (105, samples)
            if not np.isfinite(a).all():
                dropped["non_finite"] += 1
                continue
            xs.append(_prepare_window(a, decimate_factor, n_times))
            idx.append(i)

    x = np.stack(xs) if xs else np.zeros((0, N_CHANNELS, n_times), dtype=np.float32)
    rec = SubjectRecordings(code, sentences, x, np.asarray(idx, dtype=np.int64), dropped)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, x=rec.x, sentence_idx=rec.sentence_idx, sentences=np.array(sentences, dtype=str),
                     dropped=json.dumps(dropped))
        tmp.replace(cache)
    return rec


def find_subject_files(root: str | Path) -> dict[str, Path]:
    files = {}
    for p in sorted(Path(root).rglob("results*_NR.mat")):
        m = _NAME.match(p.name)
        if m:
            files[m.group("code")] = p
    if not files:
        raise FileNotFoundError(
            f"no results*_NR.mat files under {root}; see data/README.md for how to get ZuCo 2.0"
        )
    return files


def load_zuco2_nr(
    root: str | Path,
    *,
    subjects: list[str] | None = None,
    max_subjects: int | None = None,
    decimate_factor: int = 5,
    n_times: int = 640,
    min_seconds_per_word: float = 0.05,
    text_dim: int = 64,
    cache_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> EEGData:
    """Load ZuCo 2.0 normal reading as `EEGData`.

    Defaults: 500 Hz / 5 = 100 Hz, windows of 640 samples = 6.4 s (longer sentences are
    cropped to their first 6.4 s, shorter ones zero-padded at the end). Sentence id =
    position in the file (identical across people). One recording per person per sentence.
    """
    say = progress or (lambda *_: None)
    files = find_subject_files(root)
    if subjects is not None:
        missing = set(subjects) - set(files)
        if missing:
            raise ValueError(f"subjects not found: {sorted(missing)}; available: {sorted(files)}")
        files = {c: files[c] for c in subjects}
    if max_subjects is not None:
        files = dict(list(files.items())[:max_subjects])

    recs = []
    for code, p in files.items():
        r = read_subject_file(p, decimate_factor=decimate_factor, n_times=n_times,
                              min_seconds_per_word=min_seconds_per_word, cache_dir=cache_dir)
        say(f"{code}: kept {len(r.x)}/{len(r.sentences)}  dropped {r.dropped}")
        recs.append(r)

    sentences = recs[0].sentences
    for r in recs[1:]:
        if r.sentences != sentences:
            first = next(i for i, (a, b) in enumerate(zip(r.sentences, sentences)) if a != b) \
                if len(r.sentences) == len(sentences) else -1
            raise ValueError(f"{r.code} has different sentences than {recs[0].code} (first difference at {first})")

    if sum(len(r.x) for r in recs) == 0:
        raise ValueError("no usable recordings after quality filtering")
    return EEGData(
        x=np.concatenate([r.x for r in recs]),
        subject_ids=np.concatenate([np.full(len(r.x), s, dtype=np.int64) for s, r in enumerate(recs)]),
        sentence_ids=np.concatenate([r.sentence_idx for r in recs]),
        text_emb=tfidf_svd_embeddings(sentences, text_dim),
        sentences=list(sentences),
        subject_codes=[r.code for r in recs],
        sfreq=SFREQ / decimate_factor,
        meta={
            "dropped": {r.code: r.dropped for r in recs},
            "kept": {r.code: int(len(r.x)) for r in recs},
            "source_sfreq": SFREQ,
            "decimate_factor": decimate_factor,
            "n_times": n_times,
            "window_seconds": n_times * decimate_factor / SFREQ,
            "min_seconds_per_word": min_seconds_per_word,
        },
    )
