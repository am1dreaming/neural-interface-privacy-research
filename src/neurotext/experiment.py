"""One full experiment for one seed: train, evaluate against baselines, calibration curve,
privacy probe, efficiency. Used by experiments/run.py and experiments/run_seeds.py.

With `source: synthetic` the result checks the pipeline; it is not a research result.
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
import torch

from .data import (
    check_no_leakage,
    load_eegmmidb,
    load_zuco2_nr,
    make_synthetic,
    select_calibration,
    subject_and_sentence_split,
)
from .efficiency import (
    count_parameters,
    estimate_int8_linear_bytes,
    measure_latency_ms,
    quantized_size_bytes,
    serialized_size_bytes,
)
from .classical import position_oracle
from .eval import run_baselines
from .eval.baselines import signal_gain
from .models import EEGEncoder, TextProjector
from .privacy import identity_leakage
from .training import embed, fit_adapter, fit_alignment, project_text


_REAL_DATA: dict[str, object] = {}  # real datasets are slow to load: keep them for the whole process


def load_data(cfg: dict):
    """Dispatch on `cfg["source"]`. Every source returns an object with x, subject_ids,
    sentence_ids and text_emb (`EEGData`, or `SyntheticData` which has the same core fields)."""
    source = cfg["source"]
    kwargs = {k: v for k, v in cfg.items() if k != "source"}
    if source == "synthetic":
        return make_synthetic(**kwargs)
    if source == "zuco2_nr":
        kwargs.pop("seed", None)  # loading is deterministic; the seed only affects splits and initialisation
        key = json.dumps({"source": source, **kwargs}, sort_keys=True, default=str)
        if key not in _REAL_DATA:
            _REAL_DATA[key] = load_zuco2_nr(**kwargs, progress=print)
        return _REAL_DATA[key]
    if source == "eegmmidb":
        kwargs.pop("seed", None)
        key = json.dumps({"source": source, **kwargs}, sort_keys=True, default=str)
        if key not in _REAL_DATA:
            _REAL_DATA[key] = load_eegmmidb(**kwargs, progress=print)
        return _REAL_DATA[key]
    raise NotImplementedError(f"data source {source!r} has no loader yet; see data/README.md")


def run_once(
    cfg: dict, seed: int, log: Callable[[str], None] | None = None, include_efficiency: bool = True
) -> dict:
    say = log or (lambda *_: None)

    data = load_data({**cfg["data"], "seed": seed})
    split = subject_and_sentence_split(data.subject_ids, data.sentence_ids, seed=seed, **cfg["split"])
    check_no_leakage(split, data.subject_ids, data.sentence_ids)
    say(f"samples: train={len(split.train)} val={len(split.val)} test={len(split.test)}  (no leakage)")

    n_ch, n_t = data.x.shape[1], data.x.shape[2]
    m = cfg["model"]
    torch.manual_seed(seed)  # initial weights must depend on the seed, or runs are not reproducible
    encoder = EEGEncoder(n_ch, n_t, **m)
    projector = TextProjector(data.text_emb.shape[1], m["out_dim"])

    history = fit_alignment(
        encoder, projector, data.x[split.train], data.sentence_ids[split.train], data.text_emb,
        seed=seed, **cfg["train"],
    )
    say(f"alignment loss: {history[0]:.3f} -> {history[-1]:.3f}")

    # candidates = unseen test sentences, as the model sees them in the shared space
    test_sentences = split.groups["test_sentences"]
    cand = project_text(projector, data.text_emb[test_sentences])
    to_cand = {s: i for i, s in enumerate(test_sentences)}

    def evaluate(model, idx):
        y = np.array([to_cand[s] for s in data.sentence_ids[idx]])
        return run_baselines(lambda a: embed(model, a), data.x[idx], y, cand, seed=seed)

    zero = evaluate(encoder, split.test)
    # no brain data at all: told only where in the presentation order the true sentence sits.
    # Meaningful when sentence ids follow the order of presentation (ZuCo); chance on synthetic data.
    truth = np.array([to_cand[s] for s in data.sentence_ids[split.test]])
    zero["position_oracle"] = position_oracle(
        split.groups["train_sentences"], split.groups["val_sentences"], test_sentences, data.text_emb, truth)
    gain = signal_gain(zero)
    say("\n== new people, unseen sentences (zero calibration) ==")
    for name, scores in zero.items():
        say(f"  {name:16s} {({k: round(v, 3) for k, v in scores.items()})}")
    say(f"  signal gain over best baseline (top1): {gain:+.3f}")

    say("\n== calibration curve (per new person, mean over test people) ==")
    pool = np.concatenate([split.groups["train_sentences"], split.groups["val_sentences"]])
    p = cfg["personalization"]
    curve = {}
    for n_cal in p["calibration_sentences"]:
        top1s = []
        for subj in split.groups["test_subjects"]:
            eval_idx = split.test[data.subject_ids[split.test] == subj]
            cal_idx = select_calibration(data.subject_ids, data.sentence_ids, subj, pool, n_cal, seed=seed)
            assert not set(data.sentence_ids[cal_idx]) & set(data.sentence_ids[eval_idx]), "calibration leaks into eval"
            adapted = fit_adapter(
                encoder, projector, data.x[cal_idx], data.sentence_ids[cal_idx], data.text_emb,
                rank=p["rank"], epochs=p["epochs"], lr=p["lr"], seed=seed,
            )
            top1s.append(evaluate(adapted, eval_idx)["model"]["top1"])
        curve[int(n_cal)] = float(np.mean(top1s))
        say(f"  {n_cal:2d} calibration sentences -> top1 {curve[int(n_cal)]:.3f}")

    say("\n== privacy: can a linear probe tell who it is? ==")
    z = embed(encoder, data.x[split.test])
    leak = identity_leakage(z, data.subject_ids[split.test], groups=data.sentence_ids[split.test], seed=seed)
    say(f"   {({k: round(v, 3) for k, v in leak.items()})}")

    eff = None
    if include_efficiency:
        say("\n== efficiency ==")
        eff = {
            "parameters": count_parameters(encoder),
            "fp32_bytes": serialized_size_bytes(encoder),
            "int8_linear_bytes_measured": quantized_size_bytes(encoder),  # None if no quantization engine
            "int8_linear_bytes_estimated": estimate_int8_linear_bytes(encoder),
            "latency_ms_batch1_cpu": measure_latency_ms(encoder, data.x[:1]),
        }
        if eff["int8_linear_bytes_measured"] is None:
            say("   (no quantization engine in this PyTorch build: using the analytic estimate only)")
        say(f"   {eff}")

    return {
        "seed": seed,
        "synthetic": cfg["data"]["source"] == "synthetic",
        "n_samples": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "n_test_subjects": int(len(split.groups["test_subjects"])),
        "n_candidates": int(len(test_sentences)),
        "alignment_loss": [history[0], history[-1]],
        "zero_shot": zero,
        "signal_gain_top1": gain,
        "calibration_curve_top1": curve,
        "identity_leakage": leak,
        "efficiency": eff,
    }
