"""Why do adapters hurt? Compare adapter variants against zero calibration, per seed, paired.

    python experiments/adapter_sweep.py --config configs/smoke_synthetic.yaml --seeds 10

For every seed the base model is trained once, then each adapter variant is fitted on the
same calibration data and scored on the same held-out sentences of the same new people.
The reported number is (adapted top-1 - zero-calibration top-1), with a 95% interval over
seeds. Hypotheses under test:

  H1  too many / too large steps on a handful of recordings (lr, epochs)
  H2  one component is responsible (channel scale/bias vs low-rank residual)
  H3  nothing pulls the adapter back to the identity (identity regularisation)
  H4  contrastive loss over only 2-8 in-batch sentences is too easy; use all known
      sentence texts as negatives instead (negative_pool)

Synthetic data only: this probes the method, it does not say anything about real EEG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from neurotext.data import check_no_leakage, select_calibration, subject_and_sentence_split
from neurotext.eval import topk_retrieval
from neurotext.experiment import load_data
from neurotext.models import EEGEncoder, TextProjector
from neurotext.stats import mean_ci, paired_diff_ci, verdict
from neurotext.training import embed, fit_adapter, fit_alignment, project_text

VARIANTS = {
    "A current (lr1e-2, 40ep, both)":        dict(lr=1e-2, epochs=40),
    "B lower lr (1e-3, 40ep)":               dict(lr=1e-3, epochs=40),
    "C lower lr, fewer ep (1e-3, 10ep)":     dict(lr=1e-3, epochs=10),
    "D channel only":                        dict(lr=1e-2, epochs=40, components=("channel",)),
    "E low-rank only":                       dict(lr=1e-2, epochs=40, components=("lowrank",)),
    "F identity reg (1.0)":                  dict(lr=1e-2, epochs=40, identity_reg=1.0),
    "G all-text negatives":                  dict(lr=1e-2, epochs=40, pool=True),
    "H all-text negatives, lr1e-3":          dict(lr=1e-3, epochs=40, pool=True),
    "I channel mixing only":                 dict(lr=1e-2, epochs=40, components=("mixing",)),
    "J channel mixing + identity reg (1.0)": dict(lr=1e-2, epochs=40, components=("mixing",), identity_reg=1.0),
    "K channel mixing, lr1e-3":              dict(lr=1e-3, epochs=40, components=("mixing",)),
    "L all parts + identity reg (1.0)":      dict(lr=1e-2, epochs=40, components=("channel", "lowrank", "mixing"), identity_reg=1.0),
}


def run_seed(cfg: dict, seed: int, n_cals: list[int]) -> dict:
    data = load_data({**cfg["data"], "seed": seed})
    split = subject_and_sentence_split(data.subject_ids, data.sentence_ids, seed=seed, **cfg["split"])
    check_no_leakage(split, data.subject_ids, data.sentence_ids)

    m = cfg["model"]
    torch.manual_seed(seed)
    encoder = EEGEncoder(data.x.shape[1], data.x.shape[2], **m)
    projector = TextProjector(data.text_emb.shape[1], m["out_dim"])
    fit_alignment(encoder, projector, data.x[split.train], data.sentence_ids[split.train], data.text_emb,
                  seed=seed, **cfg["train"])

    test_sentences = split.groups["test_sentences"]
    cand = project_text(projector, data.text_emb[test_sentences])
    to_cand = {s: i for i, s in enumerate(test_sentences)}
    pool = np.concatenate([split.groups["train_sentences"], split.groups["val_sentences"]])  # no test text

    def top1(model, idx):
        y = np.array([to_cand[s] for s in data.sentence_ids[idx]])
        return topk_retrieval(embed(model, data.x[idx]) @ cand.T, y, (1,))["top1"]

    p = cfg["personalization"]
    subjects = split.groups["test_subjects"]
    eval_idx = {s: split.test[data.subject_ids[split.test] == s] for s in subjects}
    zero = float(np.mean([top1(encoder, eval_idx[s]) for s in subjects]))

    out = {"zero": zero, "variants": {}}
    for name, kw in VARIANTS.items():
        kw = dict(kw)
        use_pool = kw.pop("pool", False)
        out["variants"][name] = {}
        for n_cal in n_cals:
            accs = []
            for s in subjects:
                cal = select_calibration(data.subject_ids, data.sentence_ids, s, pool, n_cal, seed=seed)
                assert not set(data.sentence_ids[cal]) & set(data.sentence_ids[eval_idx[s]])
                adapted = fit_adapter(encoder, projector, data.x[cal], data.sentence_ids[cal], data.text_emb,
                                      rank=p["rank"], seed=seed, negative_pool=pool if use_pool else None, **kw)
                accs.append(top1(adapted, eval_idx[s]))
            out["variants"][name][n_cal] = float(np.mean(accs))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0, help="first seed (use fresh seeds to confirm a finding)")
    parser.add_argument("--n-cal", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--only", nargs="+", help="variant letters to run, e.g. --only A F I J")
    args = parser.parse_args()
    if args.only:
        wanted = {w.upper() for w in args.only}
        for name in [n for n in VARIANTS if n.split()[0] not in wanted]:
            del VARIANTS[name]
        if not VARIANTS:
            parser.error("--only matched no variants")
    cfg = yaml.safe_load(Path(args.config).read_text())
    print("NOTE: synthetic data: this probes the method, not real EEG.\n", flush=True)

    runs = []
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        runs.append(run_seed(cfg, seed, args.n_cal))
        print(f"seed {seed} done (zero-calibration top1 {runs[-1]['zero']:.3f})", flush=True)

    print(f"\n==== adapted minus zero-calibration top-1, mean [95% CI] over {args.seeds} seeds ====")
    print("(+ = better than no calibration, - = worse, 0 = no evidence of a difference)\n")
    zero = [r["zero"] for r in runs]
    summary = {"zero_calibration_top1": mean_ci(zero), "variants": {}}
    for name in VARIANTS:
        summary["variants"][name] = {}
        print(name)
        for n_cal in args.n_cal:
            d = paired_diff_ci([r["variants"][name][n_cal] for r in runs], zero)
            v = verdict(d)
            mark = {"better": "+", "worse": "-"}.get(v, "0")
            summary["variants"][name][n_cal] = {**d, "verdict": v}
            print(f"   {n_cal:2d} sentences: {d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]  {mark}")

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"adapter_sweep_{Path(args.config).stem}_s{args.start_seed}.json"
    path.write_text(json.dumps({"synthetic": True, "summary": summary, "runs": runs}, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
