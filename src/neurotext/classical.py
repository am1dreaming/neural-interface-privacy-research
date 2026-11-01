"""Classical baseline: ridge regression from sentence-level features to the text embedding.

Two designs, both scored by retrieval of the right sentence among the held-out sentences,
with the same split code and the same baselines as the deep model:

  cross_person   train on some people and sentences, test on NEW people reading NEW sentences
                 (the hard setting the deep model was tested in);
  within_person  one model per person, trained on that person's own recordings of the
                 training sentences, tested on the same person's held-out sentences
                 (easier: no between-person variability to cope with).

Features are z-scored per person (each person's mean/std over their own recordings, which
uses the new person's unlabeled features but never any text or labels).

`plant_signal` adds a known text-dependent pattern of chosen strength to the features. It is
a sensitivity check: if the pipeline recovers a planted signal of strength s but finds
nothing in the real data, then the real data holds less than roughly s of that kind of
signal. A null without this check says little about how weak a signal could hide in it.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from .data import EEGData, check_no_leakage, sentence_split, subject_and_sentence_split
from .eval import run_baselines
from .eval.baselines import signal_gain

ALPHAS = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def standardize_per_subject(x: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
    out = np.empty(x.shape, dtype=np.float32)
    for s in np.unique(subject_ids):
        m = subject_ids == s
        sd = x[m].std(axis=0)
        out[m] = (x[m] - x[m].mean(axis=0)) / np.where(sd > 0, sd, 1.0)
    return out


def plant_signal(x: np.ndarray, sentence_ids: np.ndarray, text_emb: np.ndarray, strength: float, seed: int = 0):
    """x + strength * (text embedding mapped through a fixed random matrix). On z-scored
    features, `strength` is the amplitude of the planted part relative to the noise (1 feature,
    same mapping for every person)."""
    if strength == 0:
        return x
    w = np.random.default_rng(10_000 + seed).standard_normal((text_emb.shape[1], x.shape[1])).astype(np.float32)
    return (x + strength * (text_emb[sentence_ids] @ w)).astype(np.float32)


def _fit_ridge(x_tr, y_tr, x_va, y_va, alphas):
    """Pick alpha by mean cosine similarity to the true text embedding on validation data."""
    best = None
    for a in alphas:
        model = Ridge(alpha=a).fit(x_tr, y_tr)
        score = float((_unit(model.predict(x_va)) * _unit(y_va)).sum(axis=1).mean())
        if best is None or score > best[0]:
            best = (score, a, model)
    return best[2], best[1]


def position_oracle(train_sentences, val_sentences, test_sentences, text_emb, truth,
                    ks=(1, 2, 4), top=(1, 5)) -> dict:
    """"Oracle position" baseline: uses NO brain data, only where the true sentence sits in the
    presentation order. The prediction for a sentence is the mean text embedding of its k nearest
    TRAINING sentences in time. k is chosen on the validation sentences (never on the test ones).

    It approximates what any model could gain from recognising "roughly when in the session this
    recording was made" (topic blocks plus slow drifts), because it is told the position exactly.
    A model that does not beat it has not shown evidence of reading language.

    `truth[j]` is the index into `test_sentences` of recording j, so the score is per recording,
    exactly like the model's."""
    from .eval import topk_retrieval

    train = np.sort(np.asarray(train_sentences))

    def predict(sentences, k):
        k = min(k, len(train))
        return _unit(np.stack([text_emb[train[np.argsort(np.abs(train - s), kind="stable")[:k]]].mean(axis=0)
                               for s in sentences]))

    val = np.asarray(val_sentences)
    best_k = ks[0]
    if len(val) >= 2:
        val_cand = _unit(text_emb[val])
        val_scores = [topk_retrieval(predict(val, k) @ val_cand.T, np.arange(len(val)), (1,))["top1"] for k in ks]
        best_k = ks[int(np.argmax(val_scores))]  # ties -> smallest k
    test_sentences = np.asarray(test_sentences)
    sim = predict(test_sentences, best_k) @ _unit(text_emb[test_sentences]).T
    return {**topk_retrieval(sim[np.asarray(truth)], truth, top), "k": best_k}


def _prepare(data: EEGData, plant: float, seed: int):
    x = standardize_per_subject(data.x, data.subject_ids)
    return plant_signal(x, data.sentence_ids, data.text_emb, plant, seed), data.text_emb[data.sentence_ids]


def cross_person(data: EEGData, seed: int, val_frac: float = 0.15, test_frac: float = 0.25,
                 alphas=ALPHAS, plant: float = 0.0, block_size: int | None = None) -> dict:
    split = subject_and_sentence_split(data.subject_ids, data.sentence_ids, val_frac, test_frac, seed, block_size)
    check_no_leakage(split, data.subject_ids, data.sentence_ids)
    x, y = _prepare(data, plant, seed)
    model, alpha = _fit_ridge(x[split.train], y[split.train], x[split.val], y[split.val], alphas)

    test_sentences = split.groups["test_sentences"]
    cand = _unit(data.text_emb[test_sentences])
    to_idx = {s: i for i, s in enumerate(test_sentences)}
    truth = np.array([to_idx[s] for s in data.sentence_ids[split.test]])
    scores = run_baselines(lambda a: _unit(model.predict(a)), x[split.test], truth, cand, seed=seed)
    scores["position_oracle"] = position_oracle(split.groups["train_sentences"], split.groups["val_sentences"], test_sentences,
                                                data.text_emb, truth)
    return {"scores": scores, "gain_top1": signal_gain(scores), "gain_top5": signal_gain(scores, "top5"),
            "alpha": alpha, "n_test": int(len(split.test)), "n_candidates": int(len(test_sentences))}


def within_person(data: EEGData, seed: int, val_frac: float = 0.15, test_frac: float = 0.25,
                  alphas=ALPHAS, plant: float = 0.0, min_samples: int = 20, block_size: int | None = None) -> dict:
    split = sentence_split(data.sentence_ids, val_frac, test_frac, seed, block_size)
    check_no_leakage(split, data.subject_ids, data.sentence_ids, by=("sentence",))
    x, y = _prepare(data, plant, seed)

    test_sentences = split.groups["test_sentences"]
    cand = _unit(data.text_emb[test_sentences])
    to_idx = {s: i for i, s in enumerate(test_sentences)}

    per_person = []
    for s in np.unique(data.subject_ids):
        tr, va, te = (idx[data.subject_ids[idx] == s] for idx in (split.train, split.val, split.test))
        if len(tr) < min_samples or len(va) < 5 or len(te) < 5:
            continue
        model, _ = _fit_ridge(x[tr], y[tr], x[va], y[va], alphas)
        truth = np.array([to_idx[t] for t in data.sentence_ids[te]])
        res = run_baselines(lambda a: _unit(model.predict(a)), x[te], truth, cand, seed=seed + int(s))
        res["position_oracle"] = position_oracle(split.groups["train_sentences"], split.groups["val_sentences"], test_sentences,
                                                data.text_emb, truth)
        per_person.append(res)
    if not per_person:
        raise ValueError("no person had enough recordings")

    scores = {name: {k: float(np.mean([p[name][k] for p in per_person])) for k in per_person[0][name]}
              for name in per_person[0]}
    return {"scores": scores, "gain_top1": signal_gain(scores), "gain_top5": signal_gain(scores, "top5"),
            "n_people": len(per_person), "n_candidates": int(len(test_sentences))}
