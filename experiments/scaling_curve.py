"""How does identity leakage change with the number of people?

    python experiments/scaling_curve.py --config configs/eegmmidb.yaml --seeds 10                 # development pool
    python experiments/scaling_curve.py --config configs/eegmmidb_final.yaml --final --all-people  # once, at the end

Per seed the people are shuffled. The first `--n-train` of them train the encoder (development mode); the others
are the NEW people. For n in `--n-list` the first n of them (nested samples, so the points of one curve are paired)
are attacked: closed-set identification (linear / MLP probe, accuracy against chance 1/n, trained on some runs and
tested on others) and open-set verification (EER / AUC of the cosine and LDA scores; LDA is fitted on the people NOT
in the sample). Compared representations: trained encoder, the same encoder untrained, spectral features, and
their PCA to the embedding size.

Closed-set accuracy is expected to FALL as n grows (more candidates). What the curve shows is whether it stays above
1/n and whether EER stays below 0.5.

Final mode (`--final`): uses the pre-registered held-out people TOO (data.holdout must be 'include'), no encoder is
trained (features only), the output is `results/final_scaling.json`, and the script refuses to overwrite it.
Pre-registered (H2, development mode): at every n, linear-probe accuracy of the trained encoder is above 1/n and
its LDA EER below 0.5 (one-sided t-tests over seeds, Holm over 2 x len(n_list)).
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import yaml

from neurotext.audit import clean, fmt, guard_single_run, safe_ci, save_json, sfreq_of
from neurotext.experiment import load_data
from neurotext.features import feature_table
from neurotext.leakage import evaluate_identity
from neurotext.privacy.probes import PROBES
from neurotext.privacy.verification import SCORERS
from neurotext.stats import holm, one_sided_p


def pca_projection(feats: np.ndarray, fit_mask: np.ndarray, dim: int) -> np.ndarray:
    """PCA of the z-scored features, fitted on the training people only."""
    mu, sd = feats[fit_mask].mean(0), feats[fit_mask].std(0)
    sd[sd == 0] = 1.0
    z = (feats - mu) / sd
    _, _, vt = np.linalg.svd(z[fit_mask], full_matrices=False)
    return (z @ vt[:dim].T).astype(np.float32)


def train_encoders(cfg, data, train_mask, seed):
    import copy as _copy

    import torch

    from neurotext.models import TextProjector, build_encoder, model_options
    from neurotext.training import embed, fit_alignment

    name, opts = model_options(cfg["model"])
    torch.manual_seed(seed)
    enc = build_encoder(name, data.x.shape[1], data.x.shape[2], **opts)
    untrained = _copy.deepcopy(enc)
    proj = TextProjector(data.text_emb.shape[1], cfg["model"]["out_dim"])
    hist = fit_alignment(enc, proj, data.x[train_mask], data.sentence_ids[train_mask], data.text_emb,
                         seed=seed, **cfg["train"])
    return {"trained encoder": embed(enc, data.x), "untrained encoder": embed(untrained, data.x)}, hist


def run_seed(cfg, data, feats, seed, n_list, n_train, probes, verify, final):
    rng = np.random.default_rng(seed)
    people = rng.permutation(np.unique(data.subject_ids))
    block = cfg["split"].get("block_size") or 1
    if final:
        train_people, pool = people[:0], people
        reps, loss = {"band power (all features)": feats}, None
    else:
        train_people, pool = people[:n_train], people[n_train:]
        train_mask = np.isin(data.subject_ids, train_people)
        reps, hist = train_encoders(cfg, data, train_mask, seed)
        loss = [hist[0], hist[-1]]
        reps["band power (all features)"] = feats
        reps["band power, PCA to embedding size"] = pca_projection(feats, train_mask, cfg["model"]["out_dim"])
    results = {}
    for n in n_list:
        if n > len(pool):
            continue
        new = np.isin(data.subject_ids, pool[:n])
        results[str(n)] = evaluate_identity(reps, data.subject_ids, data.sentence_ids, new, block, probes, seed,
                                            populations=("new_users",), verify=verify)["new_users"]
        t = results[str(n)].get("trained encoder", results[str(n)]["band power (all features)"])[probes[0]]
        print(f"seed {seed:2d} n={n:3d}: {probes[0]} probe accuracy "
              f"{'n/a' if not t else format(t['accuracy'], '.3f')} (chance {1 / n:.3f})", flush=True)
    return {"seed": seed, "n_train_people": int(len(train_people)), "alignment_loss": loss, "results": results}


def summarise(runs, probes, n_list):
    reps = list(next(iter(runs[0]["results"].values())))
    out = {"n_list": [n for n in n_list if str(n) in runs[0]["results"]], "reps": {}, "verify": {}}
    for rep in reps:
        out["reps"][rep], out["verify"][rep] = {}, {}
        for p in probes:
            out["reps"][rep][p] = {}
            for n in out["n_list"]:
                cells = [r["results"][str(n)][rep][p] for r in runs if str(n) in r["results"]]
                out["reps"][rep][p][str(n)] = {
                    "accuracy": safe_ci([c["accuracy"] if c else None for c in cells]),
                    "excess": safe_ci([c["excess_over_chance"] if c else None for c in cells]),
                    "chance": 1.0 / n}
        for s in SCORERS:
            out["verify"][rep][s] = {}
            for n in out["n_list"]:
                cells = [(r["results"][str(n)][rep].get("verify") or {}).get(s) for r in runs if str(n) in r["results"]]
                out["verify"][rep][s][str(n)] = {k: safe_ci([c[k] if c else None for c in cells]) for k in ("eer", "auc")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-list", nargs="+", type=int, default=[5, 10, 20, 40, 64])
    ap.add_argument("--n-train", type=int, default=20, help="people who train the encoder (development mode)")
    ap.add_argument("--probes", nargs="+", default=list(PROBES), choices=list(PROBES))
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--final", action="store_true", help="pre-registered final run on all people, features only; once")
    ap.add_argument("--all-people", action="store_true", help="add the point 'everybody in the data' to the n list")
    ap.add_argument("--rerun-because", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.seeds < 3:
        ap.error("use at least 3 seeds")
    cfg = yaml.safe_load(Path(args.config).read_text())
    source, hold = cfg["data"]["source"], cfg["data"].get("holdout", "exclude")
    if source == "eegmmidb" and args.final != (hold == "include"):
        ap.error("development mode needs data.holdout: exclude; --final needs data.holdout: include")
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    out_path = Path(args.out or Path(cfg.get("output_dir", "results")) / ("final_scaling.json" if args.final else "scaling_curve.json"))
    if args.final:
        guard_single_run(out_path, args.rerun_because)

    data = load_data({**cfg["data"], "seed": 0})
    feats = feature_table(data.x, sfreq_of(data, cfg)).x
    n_people = len(np.unique(data.subject_ids))
    n_list = sorted(set(args.n_list + ([n_people] if args.all_people else [])))
    print(f"data: {len(data.x)} recordings, {n_people} people; n list {n_list}\n")

    runs = [run_seed(cfg, data, feats, s, n_list, args.n_train, tuple(args.probes), not args.no_verify, args.final)
            for s in range(args.seeds)]
    summary = summarise(runs, args.probes, n_list)

    print("\n== accuracy [95% CI over seeds] (chance = 1/n) ==")
    for rep, d in summary["reps"].items():
        print(f"\n  {rep}")
        for p in args.probes:
            print(f"    {p:7s}" + "".join(f"  n={n}: {fmt(d[p][str(n)]['accuracy'])}" for n in summary["n_list"]))
    print("\n== open-set verification, LDA score: EER [95% CI] (chance 0.5) ==")
    for rep, d in summary["verify"].items():
        print(f"  {rep:36s}" + "".join(f"  n={n}: {fmt(d['lda'][str(n)]['eer'])}" for n in summary["n_list"]))

    if not args.final and "trained encoder" in summary["reps"] and "linear" in args.probes:
        pv = {}
        for n in summary["n_list"]:
            exc = clean([(r["results"][str(n)]["trained encoder"]["linear"] or {}).get("excess_over_chance")
                         for r in runs if str(n) in r["results"]])
            eer = clean([((r["results"][str(n)]["trained encoder"].get("verify") or {}).get("lda") or {}).get("eer")
                         for r in runs if str(n) in r["results"]])
            pv[f"n={n} accuracy>1/n"] = one_sided_p(exc, 0.0, "greater")
            pv[f"n={n} LDA EER<0.5"] = one_sided_p([e - 0.5 for e in eer], 0.0, "less") if eer else float("nan")
        adj = holm(pv)
        synth = source == "synthetic"
        prereg = (not synth and args.seeds == 10 and not args.epochs and sorted(n_list) == [5, 10, 20, 40, 64]
                  and args.n_train == 20)
        kind = "pipeline_check" if synth else ("confirmatory" if prereg else "probe")
        key = f"{kind}_H2"
        title = {"pipeline_check": "PIPELINE CHECK on synthetic data, not evidence for H2",
                 "probe": "PROBE, not the pre-registered configuration (10 seeds, default epochs, n = 5 10 20 40 64, n-train 20), not evidence for H2",
                 "confirmatory": "H2 (pre-registered)"}[kind]
        yes, no = {"pipeline_check": ("pass (synthetic)", "fail (synthetic)"), "probe": ("pass (probe)", "fail (probe)"),
                   "confirmatory": ("CONFIRMED", "not confirmed")}[kind]
        summary[key] = {k: {"p": pv[k], "p_holm": adj[k], "confirmed": bool(adj[k] < 0.05)} for k in pv}
        print("\n== " + title + ", trained encoder: one-sided t-tests over seeds, Holm over the family ==")
        for k, v in summary[key].items():
            print(f"  {k:22s} p={v['p']:.4f}  Holm p={v['p_holm']:.4f}  {yes if v['confirmed'] else no}")
        ok = all(v["confirmed"] for v in summary[key].values())
        print(f"\nH2 overall: {yes if ok else no + ' (at least one test failed)'}")
    save_json(out_path, {"config": cfg, "args": vars(args), "final": args.final, "summary": summary, "runs": runs})


if __name__ == "__main__":
    main()
