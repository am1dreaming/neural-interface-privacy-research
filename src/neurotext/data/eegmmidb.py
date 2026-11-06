"""PhysioNet EEG Motor Movement/Imagery dataset (EEGMMIDB): a second dataset for the LEAKAGE side.

109 people, 64 channels, 160 Hz, 14 runs each (R01 eyes open, R02 eyes closed, R03-R14 fist / fists-and-feet
execution and imagination). There is NO TEXT here, so decoding cannot be tested; only whether embeddings
reveal who the person is. For the encoder's training objective the "content" of a recording is the
instructed run type plus a fixed run-specific component (the analogue of "the sentence that was read").
Decoding quality on it is not measured and not claimed.

What goes into `EEGData`:
  x             (N, 64, n_times): up to `max_windows_per_run` non-overlapping windows from every run
  sentence_ids  the RUN (0..13). Splits by "sentence" are therefore splits by run, so windows of one run
                never sit on both sides of a split (the analogue of the block split used for ZuCo)
  text_emb      (14, 22) L2-normalised: one-hot run type + a fixed random run component of length 0.5

Held-out people. 24 of the 109 are reserved by a rule fixed in PREREGISTRATION.md (`HOLDOUT`). The loader
EXCLUDES them unless `holdout="include"` or `"only"` is passed, so development code cannot touch them by
accident. Runs that are missing, at another sampling rate, with another channel count or too short are
dropped and counted in `meta["dropped"]`; a person with fewer than `min_runs` usable runs is dropped.
A pure-numpy EDF reader is used so that no extra dependency is needed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .base import EEGData

ALL_CODES = tuple(f"S{i:03d}" for i in range(1, 110))
HOLDOUT_SEED = 20261105
# numpy.random.default_rng(HOLDOUT_SEED).permutation(109)[:24], sorted; fixed here so a change in numpy's
# generator can never move a person between groups (tests/test_eegmmidb.py checks that they agree today)
HOLDOUT = ("S002", "S005", "S006", "S012", "S017", "S019", "S022", "S024", "S029", "S032", "S038", "S040",
           "S066", "S067", "S079", "S080", "S081", "S083", "S088", "S096", "S099", "S100", "S102", "S108")

TYPE_NAMES = ("eyes_open", "eyes_closed", "fist_exec", "fist_imag", "fists_feet_exec", "fists_feet_imag")
RUN_TYPE = {1: 0, 2: 1, **{r: 2 for r in (3, 7, 11)}, **{r: 3 for r in (4, 8, 12)},
            **{r: 4 for r in (5, 9, 13)}, **{r: 5 for r in (6, 10, 14)}}
N_RUNS = 14
SFREQ = 160.0
N_CHANNELS = 64


# ---------------------------------------------------------------- EDF reading

def read_edf(path) -> tuple[list[str], float, np.ndarray]:
    """Read an EDF / EDF+ file. Returns (channel labels, sampling rate, data (n_channels, n_samples) float32
    in physical units). The EDF+ annotation channel is skipped. All data channels must share one rate."""
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(256)
        if len(head) < 256 or head[:8] != b"0       ":
            raise ValueError(f"{path.name}: not an EDF file")
        header_bytes, n_records = int(head[184:192]), int(head[236:244])
        duration, ns = float(head[244:252]), int(head[252:256])
        sig = f.read(ns * 256)
    pos = 0

    def field(width: int) -> list[bytes]:
        nonlocal pos
        out = [sig[pos + k * width : pos + (k + 1) * width] for k in range(ns)]
        pos += ns * width
        return out

    labels = [b.decode("ascii", "replace").strip() for b in field(16)]
    field(80)  # transducer
    field(8)   # physical dimension
    pmin, pmax, dmin, dmax = ([float(b) for b in field(8)] for _ in range(4))
    field(80)  # prefiltering
    nsamp = [int(b) for b in field(8)]
    per_record = sum(nsamp)
    raw = np.fromfile(path, dtype="<i2", offset=header_bytes)
    if n_records < 0 or n_records * per_record > raw.size:
        n_records = raw.size // per_record  # unknown or truncated: keep the complete records
    raw = raw[: n_records * per_record].reshape(n_records, per_record)

    keep, data, rates, off = [], [], set(), 0
    for k in range(ns):
        if labels[k] != "EDF Annotations":
            seg = raw[:, off : off + nsamp[k]].reshape(-1).astype(np.float32)
            gain = (pmax[k] - pmin[k]) / (dmax[k] - dmin[k])
            data.append((seg - dmin[k]) * gain + pmin[k])
            keep.append(labels[k])
            rates.add(round(nsamp[k] / duration, 6))
        off += nsamp[k]
    if len(rates) != 1:
        raise ValueError(f"{path.name}: channels have different sampling rates {sorted(rates)}")
    return keep, float(rates.pop()), np.stack(data)


# ---------------------------------------------------------------- preprocessing

def decimate(x: np.ndarray, q: int) -> np.ndarray:
    """Zero-phase low-pass (windowed sinc, cutoff 0.9 of the new Nyquist) and keep every q-th sample."""
    if q == 1:
        return x
    n_taps = 16 * q + 1
    t = np.arange(n_taps) - n_taps // 2
    fc = 0.45 / q
    h = 2 * fc * np.sinc(2 * fc * t) * np.hamming(n_taps)
    h /= h.sum()
    length = x.shape[-1]
    nfft = 1 << int(np.ceil(np.log2(length + n_taps)))
    y = np.fft.irfft(np.fft.rfft(x, nfft, axis=-1) * np.fft.rfft(h, nfft), nfft, axis=-1)
    return y[..., n_taps // 2 : n_taps // 2 + length][..., ::q].astype(np.float32)


def window_starts(n_samples: int, win: int, max_windows: int) -> np.ndarray:
    """Start indices of up to `max_windows` non-overlapping windows spread evenly over the recording."""
    n = n_samples // win
    if n == 0:
        return np.array([], dtype=int)
    blocks = np.unique(np.linspace(0, n - 1, min(max_windows, n)).round().astype(int))
    return blocks * win


def text_embeddings(extra_dim: int = 16) -> np.ndarray:
    """(14, 6 + extra_dim) L2-normalised: one-hot run type + a fixed random run component of length 0.5. Every run
    is then its own 'sentence' (so the contrastive loss is well posed) with a type structure shared by runs."""
    rng = np.random.default_rng(0)
    emb = np.zeros((N_RUNS, len(TYPE_NAMES) + extra_dim), dtype=np.float32)
    for run in range(1, N_RUNS + 1):
        emb[run - 1, RUN_TYPE[run]] = 1.0
    extra = rng.standard_normal((N_RUNS, extra_dim))
    emb[:, len(TYPE_NAMES):] = 0.5 * extra / np.linalg.norm(extra, axis=1, keepdims=True)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def _select_codes(root: Path, holdout: str, subjects, max_subjects) -> list[str]:
    if holdout not in ("exclude", "include", "only"):
        raise ValueError("holdout must be 'exclude', 'include' or 'only'")
    wanted = list(subjects) if subjects else list(ALL_CODES)
    if holdout == "exclude":
        wanted = [c for c in wanted if c not in HOLDOUT]
    elif holdout == "only":
        wanted = [c for c in wanted if c in HOLDOUT]
    wanted = [c for c in wanted if (root / c).is_dir()]
    return wanted[:max_subjects] if max_subjects else wanted


def _load_subject(root: Path, code: str, decimate_factor: int, win: int, max_windows: int, skip: int,
                  cache: Path | None, tag: str):
    cache_file = cache / f"{code}_{tag}.npz" if cache else None
    if cache_file is not None and cache_file.exists():
        z = np.load(cache_file, allow_pickle=False)
        return z["x"], z["runs"], list(z["names"]), json.loads(str(z["dropped"]))
    xs, runs, names, dropped = [], [], [], Counter()
    for run in range(1, N_RUNS + 1):
        path = root / code / f"{code}R{run:02d}.edf"
        if not path.exists():
            dropped["missing"] += 1
            continue
        try:
            labels, fs, sig = read_edf(path)
        except (ValueError, OSError):
            dropped["unreadable"] += 1
            continue
        if abs(fs - SFREQ) > 1e-3:
            dropped["other_sampling_rate"] += 1
            continue
        if sig.shape[0] != N_CHANNELS:
            dropped["other_channel_count"] += 1
            continue
        sig = decimate(sig, decimate_factor)[:, skip:]
        starts = window_starts(sig.shape[1], win, max_windows)
        if len(starts) == 0:
            dropped["too_short"] += 1
            continue
        if not np.isfinite(sig).all() or float(sig.std()) == 0.0:
            dropped["non_finite_or_flat"] += 1
            continue
        names = names or labels
        xs.append(np.stack([sig[:, s : s + win] for s in starts]))
        runs.append(np.full(len(starts), run - 1))
    x = np.concatenate(xs).astype(np.float32) if xs else np.zeros((0, N_CHANNELS, win), np.float32)
    r = np.concatenate(runs) if runs else np.zeros(0, int)
    if cache_file is not None and len(x):
        cache.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, x=x, runs=r, names=np.array(names), dropped=json.dumps(dict(dropped)))
    return x, r, names, dict(dropped)


def load_eegmmidb(
    root: str | Path,
    cache_dir: str | Path | None = None,
    decimate_factor: int = 2,
    window_seconds: float = 4.0,
    max_windows_per_run: int = 6,
    skip_seconds: float = 1.0,
    min_runs: int = 12,
    holdout: str = "exclude",
    subjects: list[str] | None = None,
    max_subjects: int | None = None,
    progress=None,
) -> EEGData:
    """Load EEGMMIDB from `root/S001/S001R01.edf ...`.

    `window_seconds * 160 / decimate_factor` must be a whole number divisible by the encoder's patch
    (default 4 s at 80 Hz = 320 samples). `holdout`: 'exclude' (default) keeps the pre-registered held-out
    people out of the data; 'only' loads just them; 'include' loads everybody.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found; see data/README.md (EEGMMIDB) for how to download it")
    fs = SFREQ / decimate_factor
    win = int(round(window_seconds * fs))
    skip = int(round(skip_seconds * fs))
    cache = Path(cache_dir) if cache_dir else None
    tag = f"d{decimate_factor}_w{win}_k{max_windows_per_run}_s{skip}"
    codes = _select_codes(root, holdout, subjects, max_subjects)
    if not codes:
        raise FileNotFoundError(f"no subject folders (S001 ...) found in {root} for holdout={holdout!r}")

    xs, subj, sent, kept, names = [], [], [], [], []
    dropped: dict[str, dict] = {}
    dropped_subjects: dict[str, str] = {}
    for n, code in enumerate(codes, 1):
        x, runs, ch, drop = _load_subject(root, code, decimate_factor, win, max_windows_per_run, skip, cache, tag)
        dropped[code] = drop
        n_runs = len(np.unique(runs))
        if n_runs < min_runs:
            dropped_subjects[code] = f"only {n_runs} usable runs (< {min_runs})"
            continue
        names = names or ch
        xs.append(x)
        subj.append(np.full(len(x), len(kept)))
        sent.append(runs)
        kept.append(code)
        if progress:
            progress(f"  [{n}/{len(codes)}] {code}: {len(x)} windows from {n_runs} runs")
    if not kept:
        raise ValueError("no usable subject")
    return EEGData(
        x=np.concatenate(xs), subject_ids=np.concatenate(subj), sentence_ids=np.concatenate(sent),
        text_emb=text_embeddings(), sentences=[f"run {r:02d}: {TYPE_NAMES[RUN_TYPE[r]]}" for r in range(1, N_RUNS + 1)],
        subject_codes=kept, sfreq=fs,
        meta={"dropped": dropped, "dropped_subjects": dropped_subjects, "channel_names": names,
              "holdout": holdout, "holdout_codes": list(HOLDOUT), "window_seconds": window_seconds,
              "run_types": [TYPE_NAMES[RUN_TYPE[r]] for r in range(1, N_RUNS + 1)]},
    )


def holdout_ids(data: EEGData) -> np.ndarray:
    """Integer subject ids (in `data`) of the pre-registered held-out people that the data contains."""
    return np.array([i for i, c in enumerate(data.subject_codes) if c in HOLDOUT], dtype=int)
