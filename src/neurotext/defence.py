"""Defences against identity leakage from embeddings, and their price in decoding accuracy.

Four defences, each applied to the embedding before it is released:

  noise            add Gaussian noise of total norm `sigma` to the (unit-norm) embedding;
  pca-k            keep only the k strongest directions of the embeddings of the training people;
  id-null-m        remove the m strongest directions along which the TRAINING people's average
                   embeddings differ, i.e. delete the subspace a linear attacker would use;
  adversarial-l    train the encoder against an adversary that tries to name the person
                   (gradient reversal, weight l); no post-processing.

Everything that is "fitted" (PCA basis, discriminant directions, the adversary) is fitted on the
training people only, never on the new users whose leakage is then measured.

The attacker is adaptive: the identity probe is trained on the DEFENDED embeddings of the
attacker's enrolment sentences, so it knows about the noise or the compression. Utility is the
retrieval of the right sentence among unseen test sentences for new users, from the same
defended embedding, so the user pays the price of the defence in exactly the place where the
attacker is hurt.
"""

from __future__ import annotations

import numpy as np
from .data import check_no_leakage, subject_and_sentence_split
from .eval import topk_retrieval
from .leakage import evaluate_identity
from .privacy.probes import PROBES


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def noise_defence(z: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    """z + noise whose total norm is about `sigma` (z has unit norm), then renormalised."""
    rng = np.random.default_rng(10_000 + seed)
    return _unit(z + sigma / np.sqrt(z.shape[1]) * rng.standard_normal(z.shape))


def pca_defence(z_fit: np.ndarray, z: np.ndarray, k: int) -> np.ndarray:
    """Project onto the k strongest principal directions of `z_fit` (fitted on training people)."""
    mu = z_fit.mean(axis=0)
    _, _, vt = np.linalg.svd(z_fit - mu, full_matrices=False)
    v = vt[: min(k, vt.shape[0])].T
    return mu + (z - mu) @ v @ v.T


def identity_null_defence(z_fit: np.ndarray, labels_fit: np.ndarray, z: np.ndarray, m: int) -> np.ndarray:
    """Remove the m strongest directions of between-person variation found on the training people:
    the principal directions of the matrix of per-person mean embeddings. With m equal to the number
    of people minus one, all linear differences between the training people's means are gone.

    (Linear discriminant directions were tried first and rejected: they define a decision boundary,
    not the subspace the people differ in, and removing all of them still left 16% of the mean
    separation in a check, which a linear classifier amplifies back.)"""
    mu = z_fit.mean(axis=0)
    means = np.stack([z_fit[labels_fit == c].mean(axis=0) for c in np.unique(labels_fit)]) - mu
    _, _, vt = np.linalg.svd(means, full_matrices=False)
    v = vt[: min(m, vt.shape[0])].T
    return z - (z - mu) @ v @ v.T


def run_defence_seed(cfg: dict, seed: int, probes: tuple[str, ...] = PROBES, verify: bool = False) -> dict:
    """One seed: train the plain encoder (and the adversarial ones), release the embeddings under every
    defence, and measure utility and identity leakage for the new users. Needs PyTorch."""
    import torch

    from .experiment import load_data
    from .models import TextProjector, build_encoder, model_options
    from .training import embed, fit_alignment, fit_alignment_adversarial, project_text

    data = load_data({**cfg["data"], "seed": seed})
    split = subject_and_sentence_split(data.subject_ids, data.sentence_ids, seed=seed, **cfg["split"])
    check_no_leakage(split, data.subject_ids, data.sentence_ids)
    block = cfg["split"].get("block_size") or 1
    m = cfg["model"]
    enc_name, enc_opts = model_options(m)
    d = cfg.get("defence", {})
    n_ch, n_t = data.x.shape[1], data.x.shape[2]

    def build():
        torch.manual_seed(seed)  # every encoder starts from the same weights
        return build_encoder(enc_name, n_ch, n_t, **enc_opts), TextProjector(data.text_emb.shape[1], m["out_dim"])

    encoder, projector = build()
    fit_alignment(encoder, projector, data.x[split.train], data.sentence_ids[split.train], data.text_emb,
                  seed=seed, **cfg["train"])
    z0 = embed(encoder, data.x)

    released = {"none": (z0, projector)}
    for sigma in d.get("noise", []):
        released[f"noise s={sigma}"] = (noise_defence(z0, sigma, seed), projector)
    z_fit, who_fit = z0[split.train], data.subject_ids[split.train]
    for k in d.get("pca", []):
        released[f"pca k={k}"] = (pca_defence(z_fit, z0, k), projector)
    for mm in d.get("identity_null", []):
        released[f"id-null m={mm}"] = (identity_null_defence(z_fit, who_fit, z0, mm), projector)

    people = {int(s): i for i, s in enumerate(np.unique(who_fit))}
    subj_idx = np.array([people[int(s)] for s in who_fit])
    for lam in d.get("adversarial", []):
        enc_a, proj_a = build()
        fit_alignment_adversarial(enc_a, proj_a, data.x[split.train], data.sentence_ids[split.train], subj_idx,
                                  data.text_emb, adv_weight=lam, seed=seed, **cfg["train"])
        released[f"adversarial l={lam}"] = (embed(enc_a, data.x), proj_a)

    test_sentences = split.groups["test_sentences"]
    to_idx = {s: i for i, s in enumerate(test_sentences)}
    truth = np.array([to_idx[s] for s in data.sentence_ids[split.test]])
    utility = {}
    for name, (z, proj) in released.items():
        cand = _unit(project_text(proj, data.text_emb[test_sentences]))
        utility[name] = topk_retrieval(z[split.test] @ cand.T, truth, (1, 5))

    new_mask = np.isin(data.subject_ids, split.groups["test_subjects"])
    leak = evaluate_identity({n: z for n, (z, _) in released.items()}, data.subject_ids, data.sentence_ids,
                             new_mask, block, probes, seed, populations=("new_users",), verify=verify)["new_users"]
    return {
        "seed": seed,
        "n_new_users": int(len(split.groups["test_subjects"])),
        "n_candidates": int(len(test_sentences)),
        "defences": {n: {"utility": utility[n], "identity": leak[n]} for n in released},
    }
