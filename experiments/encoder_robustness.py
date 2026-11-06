"""Does the leakage depend on the architecture? Four encoders, same seeds, same splits.

    python experiments/encoder_robustness.py --config configs/zuco2_nr.yaml --seeds 10
    python experiments/encoder_robustness.py --config configs/zuco2_nr.yaml --out-dims 4 8 16 32 64 --encoders transformer
    python experiments/encoder_robustness.py --config configs/zuco2_nr.yaml --defences      # + effect of every defence

Encoders: transformer (the project's), linear, small CNN, EEGNet. For every (encoder, embedding size) and seed
the encoder is trained as usual, then the identity of NEW people is attacked from its embeddings (linear and MLP
probe, closed set; cosine and LDA verification, open set), next to the same encoder before training. Splits
depend on the seed only, so encoders are compared PAIRED by seed.

Pre-registered (PREREGISTRATION.md, H4): for each of the four encoders at 32-d, the linear and MLP probe on the
trained encoder beat chance (one-sided t-test over seeds, Holm over 4 x 2). Everything else is exploratory.
Seeds change initialisation and which people are 'new', not the data: intervals are a lower bound.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import yaml

from neurotext.audit import clean, fmt, safe_ci, save_json, sfreq_of
from neurotext.defence import run_defence_seed
from neurotext.experiment import load_data
from neurotext.features import feature_table
from neurotext.leakage import run_identity_seed
from neurotext.models import ENCODERS
from neurotext.privacy.probes import PROBES
from neurotext.privacy.verification import SCORERS
from neurotext.stats import holm, one_sided_p, paired_diff_ci, verdict

CONFIRMATORY_DIM = 32


def _probe(r, rep, probe):
    return r["identity"]["new_users"][rep].get(probe)


def _verify(r, rep, scorer, key="eer"):
    v = r["identity"]["new_users"][rep].get("verify")
    cell = v.get(scorer) if v else None
    return cell[key] if cell else None


def summarise_rep(runs, rep, probes):
    out = {"leak": {}, "verify": {}}
    for p in probes:
        cells = [_probe(r, rep, p) for r in runs]
        out["leak"][p] = {
            "accuracy": safe_ci([c["accuracy"] if c else None for c in cells]),
            "excess": safe_ci([c["excess_over_chance"] if c else None for c in cells]),
            "chance": float(np.mean(clean([c["chance"] if c else None for c in cells]) or [np.nan])),
        }
    for s in SCORERS:
        out["verify"][s] = {k: safe_ci([_verify(r, rep, s, k) for r in runs]) for k in ("eer", "auc")}
    return out


def summarise_label(label, enc, dim, runs, probes, defence_runs):
    ident = runs
    s = {"encoder": enc, "out_dim": dim, "n_seeds": len(runs)}
    tr, un = summarise_rep(ident, "trained encoder", probes), summarise_rep(ident, "untrained encoder", probes)
    s["trained"], s["untrained"] = tr["leak"], un["leak"]
    s["verify_trained"], s["verify_untrained"] = tr["verify"], un["verify"]
    s["trained_minus_untrained"] = {}
    for p in probes:
        pairs = [(_probe(r, "trained encoder", p), _probe(r, "untrained encoder", p)) for r in ident]
        pairs = [(a["excess_over_chance"], b["excess_over_chance"]) for a, b in pairs if a and b]
        if len(pairs) >= 2:
            d = paired_diff_ci([a for a, _ in pairs], [b for _, b in pairs])
            s["trained_minus_untrained"][p] = {**d, "verdict": verdict(d)}
    if defence_runs:
        s["defences"] = {}
        for name in defence_runs[0]["defences"]:
            if name == "none":
                continue
            row = {}
            for p in probes:
                a = [r["defences"][name]["identity"][p] for r in defence_runs]
                b = [r["defences"]["none"]["identity"][p] for r in defence_runs]
                pairs = [(x["accuracy"], y["accuracy"]) for x, y in zip(a, b) if x and y]
                if len(pairs) >= 2:
                    d = paired_diff_ci([x for x, _ in pairs], [y for _, y in pairs])
                    row[f"d_acc_{p}"] = {**d, "leak": {"better": "up", "worse": "down"}.get(verdict(d), "same")}
            top = [(r["defences"][name]["utility"]["top1"], r["defences"]["none"]["utility"]["top1"]) for r in defence_runs]
            row["d_top1"] = paired_diff_ci([a for a, _ in top], [b for _, b in top])
            s["defences"][name] = row
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
    ap.add_argument("--out-dims", nargs="+", type=int, default=[CONFIRMATORY_DIM])
    ap.add_argument("--probes", nargs="+", default=list(PROBES), choices=list(PROBES))
    ap.add_argument("--defences", action="store_true", help="also train and score the defences from the config (slow)")
    ap.add_argument("--no-verify", action="store_true", help="skip open-set verification")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.seeds < 3:
        ap.error("use at least 3 seeds")

    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg["data"].get("holdout", "exclude") != "exclude" and cfg["data"]["source"] == "eegmmidb":
        ap.error("this is a development experiment: data.holdout must be 'exclude' (the held-out people are for final_holdout.py)")
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    verify = not args.no_verify
    data = load_data({**cfg["data"], "seed": 0})
    ft = feature_table(data.x, sfreq_of(data, cfg))
    print(f"data: {len(data.x)} recordings, {len(np.unique(data.subject_ids))} people; features {ft.x.shape[1]}-d\n")

    probes = tuple(args.probes)
    all_runs, all_def, first = {}, {}, True
    for enc in args.encoders:
        for dim in args.out_dims:
            label = f"{enc}-{dim}"
            c = copy.deepcopy(cfg)
            c["model"] = {**cfg["model"], "name": enc, "out_dim": dim}
            all_runs[label], all_def[label] = [], []
            for seed in range(args.seeds):
                r = run_identity_seed(data, ft.x, c, seed, probes, verify=verify, feature_reps=first)
                all_runs[label].append(r)
                if args.defences:
                    all_def[label].append(run_defence_seed(c, seed, probes, verify=verify))
                t = _probe(r, "trained encoder", probes[0])
                print(f"{label:16s} seed {seed:2d}: new users, {probes[0]} probe, accuracy "
                      f"{'n/a' if not t else format(t['accuracy'], '.3f')} (chance {'n/a' if not t else format(t['chance'], '.3f')})", flush=True)
            first = False

    summary = {"encoders": {}, "probes": list(probes)}
    for enc in args.encoders:
        for dim in args.out_dims:
            label = f"{enc}-{dim}"
            summary["encoders"][label] = summarise_label(label, enc, dim, all_runs[label], probes, all_def[label])
    first_label = f"{args.encoders[0]}-{args.out_dims[0]}"
    if "band power (all features)" in all_runs[first_label][0]["identity"]["new_users"]:
        summary["reference_features"] = {
            rep: summarise_rep(all_runs[first_label], rep, probes)
            for rep in ("band power (all features)", "band power, PCA to embedding size")}

    # ---- H4 (confirmatory): trained encoder beats chance, each encoder, each probe, Holm over the family
    pvals, est = {}, {}
    for enc in args.encoders:
        label = f"{enc}-{CONFIRMATORY_DIM}"
        if label not in all_runs:
            continue
        for p in probes:
            vals = clean([(_probe(r, "trained encoder", p) or {}).get("excess_over_chance") for r in all_runs[label]])
            pvals[f"{enc}/{p}"], est[f"{enc}/{p}"] = one_sided_p(vals, 0.0, "greater"), safe_ci(vals)
    adj = holm(pvals)
    synth = cfg["data"]["source"] == "synthetic"
    prereg = (not synth and args.seeds == 10 and not args.epochs and set(args.encoders) == set(ENCODERS)
              and CONFIRMATORY_DIM in args.out_dims)
    kind = "pipeline_check" if synth else ("confirmatory" if prereg else "probe")
    h4 = f"{kind}_H4"
    title = {"pipeline_check": "PIPELINE CHECK on synthetic data, not evidence for H4",
             "probe": "PROBE, not the pre-registered configuration (needs 10 seeds, default epochs, all four encoders), not evidence for H4",
             "confirmatory": "H4 (pre-registered)"}[kind]
    yes, no = {"pipeline_check": ("pass (synthetic)", "fail (synthetic)"), "probe": ("pass (probe)", "fail (probe)"),
               "confirmatory": ("CONFIRMED", "not confirmed")}[kind]
    summary[h4] = {k: {"excess": est[k], "p": pvals[k], "p_holm": adj[k], "confirmed": bool(adj[k] < 0.05)}
                   for k in pvals}

    chance = np.mean(clean([_probe(r, "trained encoder", probes[0])["chance"] for r in all_runs[first_label]
                            if _probe(r, "trained encoder", probes[0])]))
    print(f"\n== new people, trained encoder; accuracy minus chance (chance = {chance:.3f}), mean [95% CI] over seeds ==")
    print(f"{'encoder':16s}" + "".join(f"{p + ' probe':>30s}" for p in probes) + f"{'LDA EER':>26s}")
    for label, s in summary["encoders"].items():
        print(f"{label:16s}" + "".join(f"{fmt(s['trained'][p]['excess'], True):>30s}" for p in probes)
              + f"{fmt(s['verify_trained']['lda']['eer']):>26s}")
    print("\n== paired: trained minus untrained (same architecture and initial weights) ==")
    for label, s in summary["encoders"].items():
        print(f"{label:16s}" + "".join(f"  {p}: {fmt(s['trained_minus_untrained'].get(p), True)} {s['trained_minus_untrained'].get(p, {}).get('verdict', '')}"
                                        for p in probes))
    if summary[h4]:
        print("\n== " + title + ": trained encoder above chance; one-sided t-test over seeds, Holm over the family ==")
        for k, v in summary[h4].items():
            print(f"  {k:22s} excess {fmt(v['excess'], True)}  p={v['p']:.4f}  Holm p={v['p_holm']:.4f}  "
                  f"{yes if v['confirmed'] else no}")
    save_json(Path(args.out or Path(cfg.get("output_dir", "results")) / "encoder_robustness.json"),
              {"config": cfg, "args": vars(args), "summary": summary, "runs": all_runs, "defence_runs": all_def})


if __name__ == "__main__":
    main()
