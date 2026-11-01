"""ZuCo 2.0 (normal reading): the authors' precomputed band-power features per sentence.

`sentenceData.mean_<band>` holds one value per channel for the whole sentence, for eight
bands: t1, t2 (theta), a1, a2 (alpha), b1, b2 (beta), g1, g2 (gamma). That is 8 x 105 = 840
numbers per person and sentence. They feed classical models that need no deep network.

The same quality screen as the raw-EEG loader is applied (`screen_recording`), so the
classical and the deep model are always compared on exactly the same sentences.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .base import EEGData
from .text_features import tfidf_svd_embeddings
from .zuco import N_CHANNELS, _NAME, _decode, _h5py, find_subject_files, screen_recording

BANDS = ("t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2")


@dataclass
class BandFeatures:
    code: str
    sentences: list[str]
    features: np.ndarray   # (349, 840) float32; NaN rows = sentence dropped
    dropped: dict


def read_band_features(
    path: str | Path, *, min_seconds_per_word: float = 0.05, cache_dir: str | Path | None = None
) -> BandFeatures:
    path = Path(path)
    m = _NAME.match(path.name)
    code = m.group("code") if m else path.stem
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"{code}_bands_m{min_seconds_per_word}_{path.stat().st_size}.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            return BandFeatures(code, [str(s) for s in z["sentences"]], z["features"], json.loads(str(z["dropped"])))

    h5py = _h5py()
    dropped = {"no_eeg": 0, "too_short": 0, "non_finite": 0}
    with h5py.File(path, "r") as f:
        sd = f["sentenceData"]
        n = sd["content"].shape[0]
        sentences = [_decode(f, sd["content"][i, 0]) for i in range(n)]
        feats = np.full((n, len(BANDS) * N_CHANNELS), np.nan, dtype=np.float32)
        for i in range(n):
            reason = screen_recording(f, sd, i, min_seconds_per_word)
            if reason:
                dropped[reason] += 1
                continue
            parts = []
            for b in BANDS:
                o = f[sd[f"mean_{b}"][i, 0]]
                parts.append(np.asarray(o).flatten() if isinstance(o, h5py.Dataset) else np.array([]))
            if any(p.size != N_CHANNELS for p in parts):
                dropped["no_eeg"] += 1
                continue
            v = np.concatenate(parts).astype(np.float32)
            if not np.isfinite(v).all():
                dropped["non_finite"] += 1
                continue
            feats[i] = v

    rec = BandFeatures(code, sentences, feats, dropped)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, features=feats, sentences=np.array(sentences, dtype=str), dropped=json.dumps(dropped))
        tmp.replace(cache)
    return rec


def load_zuco2_nr_bandpower(
    root: str | Path,
    *,
    subjects: list[str] | None = None,
    max_subjects: int | None = None,
    min_seconds_per_word: float = 0.05,
    text_dim: int = 64,
    cache_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> EEGData:
    """Same people, sentences and text embeddings as `load_zuco2_nr`, but `x` is the
    (N, 840) band-power table instead of (N, 105, T) recordings."""
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
        r = read_band_features(p, min_seconds_per_word=min_seconds_per_word, cache_dir=cache_dir)
        say(f"{code}: kept {int(np.isfinite(r.features).all(axis=1).sum())}/{len(r.sentences)}  dropped {r.dropped}")
        recs.append(r)

    sentences = recs[0].sentences
    for r in recs[1:]:
        if r.sentences != sentences:
            raise ValueError(f"{r.code} has different sentences than {recs[0].code}")

    xs, subj, sent = [], [], []
    for s, r in enumerate(recs):
        ok = np.flatnonzero(np.isfinite(r.features).all(axis=1))
        xs.append(r.features[ok])
        subj.append(np.full(len(ok), s, dtype=np.int64))
        sent.append(ok.astype(np.int64))
    if sum(len(x) for x in xs) == 0:
        raise ValueError("no usable recordings after quality filtering")
    return EEGData(
        x=np.concatenate(xs),
        subject_ids=np.concatenate(subj),
        sentence_ids=np.concatenate(sent),
        text_emb=tfidf_svd_embeddings(sentences, text_dim),
        sentences=list(sentences),
        subject_codes=[r.code for r in recs],
        meta={"dropped": {r.code: r.dropped for r in recs}, "bands": BANDS, "min_seconds_per_word": min_seconds_per_word},
    )
