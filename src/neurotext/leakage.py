"""How much does a representation reveal about WHO produced it?

Attacker model measured here (closed set): the attacker holds labelled embeddings of every
person from some sentences ("enrolment") and tries to name the person behind embeddings of
OTHER sentences. Whole blocks of consecutive sentences are held out, because recordings made
close in time share slow drifts and would otherwise make identification look easier than it
is for an attacker who enrols in a different session.

Representations are compared on the same recordings:
  - trained encoder           the model under study;
  - untrained encoder         the same architecture and initial weights, before training:
                              leakage that exists with no learning at all (the control);
  - band power (840-d)        the signal's own features, no network;
  - band power, PCA 32-d      same size as the embedding, so size is not the explanation.

Deliberately NOT included: band power after removing each person's own mean/std. Normalising a
person's recordings by that person's statistics needs to know whose recordings they are, which is
exactly what the attacker is trying to find out, and inside cross-validation it also makes the
held-out and training means mirror images of each other, so the probe is wrong on purpose and
accuracy falls far BELOW chance (seen: 0.000). Such a number is an artefact, not privacy.

Two populations are scored: "new_users" (people the encoder never saw in training) and
"everyone" (all people, including those the encoder was trained on).
"""

from __future__ import annotations

import copy

import numpy as np

from .data import EEGData, check_no_leakage, subject_and_sentence_split
from .privacy import identity_leakage
from .privacy.probes import PROBES
from .privacy.verification import verification


def align_by_key(raw: EEGData, bands: EEGData) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of the recordings present in BOTH tables (same person code and sentence id),
    in the order of `raw`. The two loaders apply the same quality screen, so this should keep
    almost everything; the caller can check how much was lost."""
    row_in_bands = {(bands.subject_codes[s], int(t)): i
                    for i, (s, t) in enumerate(zip(bands.subject_ids, bands.sentence_ids))}
    a, b = [], []
    for i, (s, t) in enumerate(zip(raw.subject_ids, raw.sentence_ids)):
        j = row_in_bands.get((raw.subject_codes[s], int(t)))
        if j is not None:
            a.append(i)
            b.append(j)
    return np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)


def _pca(x: np.ndarray, dim: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    z = (x - x.mean(axis=0)) / np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
    return PCA(n_components=min(dim, z.shape[1], z.shape[0] - 1), random_state=0).fit_transform(z).astype(np.float32)


def evaluate_identity(
    reps: dict[str, np.ndarray],
    subject_ids: np.ndarray,
    sentence_ids: np.ndarray,
    new_mask: np.ndarray,
    block_size: int,
    probes: tuple[str, ...] = PROBES,
    seed: int = 0,
    n_splits: int = 4,
    populations: tuple[str, ...] = ("new_users", "everyone"),
    verify: bool = False,
    n_pairs: int = 20000,
) -> dict:
    """result[population][representation][probe] = identity_leakage(...) dict (or None if the
    population is too small to score).

    With `verify=True`, result["new_users"][representation]["verify"] also holds open-set verification
    (EER / AUC of the cosine score and of the LDA score fitted on the people who are NOT new users),
    from pairs of recordings in different groups. Not computed for "everyone": the LDA would be fitted
    on the people it is then scored on."""
    out: dict = {}
    all_pops = {"new_users": np.asarray(new_mask, dtype=bool), "everyone": np.ones(len(subject_ids), bool)}
    for pop in populations:
        mask = all_pops[pop]
        out[pop] = {}
        groups = sentence_ids[mask] // block_size
        for name, x in reps.items():
            out[pop][name] = {}
            for probe in probes:
                try:
                    out[pop][name][probe] = identity_leakage(
                        x[mask], subject_ids[mask], groups=groups, n_splits=n_splits, seed=seed,
                        probe=probe, standardize=True)
                except ValueError:
                    out[pop][name][probe] = None
            if verify and pop == "new_users":
                fit = ~mask
                try:
                    out[pop][name]["verify"] = verification(
                        x[mask], subject_ids[mask], groups,
                        x[fit] if fit.any() else None, subject_ids[fit] if fit.any() else None,
                        n_pairs=n_pairs, seed=seed)
                except ValueError:
                    out[pop][name]["verify"] = None
    return out


def run_identity_seed(raw, band_x: np.ndarray, cfg: dict, seed: int, probes: tuple[str, ...] = PROBES,
                      verify: bool = False, feature_reps: bool = True) -> dict:
    """Train the encoder for one seed (same recipe as `neurotext.experiment.run_once`) and score
    identity leakage of every representation. `band_x` is the (N, F) band-power table, row-aligned
    with `raw.x`. Needs PyTorch."""
    import torch

    from .models import TextProjector, build_encoder, model_options
    from .training import embed, fit_alignment

    block = cfg["split"].get("block_size") or 1
    split = subject_and_sentence_split(raw.subject_ids, raw.sentence_ids, seed=seed, **cfg["split"])
    check_no_leakage(split, raw.subject_ids, raw.sentence_ids)

    m = cfg["model"]
    torch.manual_seed(seed)
    enc_name, enc_opts = model_options(m)
    encoder = build_encoder(enc_name, raw.x.shape[1], raw.x.shape[2], **enc_opts)
    untrained = copy.deepcopy(encoder)  # control: same architecture, same initial weights, never trained
    projector = TextProjector(raw.text_emb.shape[1], m["out_dim"])
    history = fit_alignment(encoder, projector, raw.x[split.train], raw.sentence_ids[split.train], raw.text_emb,
                            seed=seed, **cfg["train"])

    reps = {
        "trained encoder": embed(encoder, raw.x),
        "untrained encoder": embed(untrained, raw.x),
    }
    if feature_reps:  # the feature representations do not depend on the encoder: skip them when comparing encoders
        reps["band power (all features)"] = band_x.astype(np.float32)
        reps["band power, PCA to embedding size"] = _pca(band_x, m["out_dim"])
    new_mask = np.isin(raw.subject_ids, split.groups["test_subjects"])
    res = evaluate_identity(reps, raw.subject_ids, raw.sentence_ids, new_mask, block, probes, seed, verify=verify)
    return {"seed": seed, "n_new_users": int(len(split.groups["test_subjects"])),
            "alignment_loss": [history[0], history[-1]], "identity": res}
