"""Where does the leakage come from? Physiology or recording artefact.

    python experiments/feature_ablation.py --config configs/zuco2_nr.yaml --seeds 5
    python experiments/feature_ablation.py --config configs/eegmmidb.yaml --seeds 5 --channels

EXPLORATORY (PREREGISTRATION.md): leakage from spectral features on ZuCo is already known to be near the ceiling,
so no threshold is tested; the questions were fixed in advance.

Interpretable features (neurotext/features.py): log RMS per channel, log power in delta / theta / alpha / beta / gamma
per channel, individual alpha peak (frequency and power). Every setting is scored the same way as the embeddings:
closed-set identification (linear and MLP probe, held-out sentence blocks / runs; accuracy against chance 1/n) and
open-set verification (EER of the LDA score; LDA fitted on a disjoint half of the people).

Settings:
  all             every feature group;
  only X          one group alone: how much leakage it carries by itself;
  all minus X     one group removed: how much leakage is left without it;
  controls        per-RECORDING amplitude normalisation (every channel of every recording divided by its own
                  std), trimmed window edges (10% off both ends), both together. If leakage survives them,
                  it is not the amplitude of the recording or the start / end of the window.
Never used: normalising by a person's own statistics (it uses the label, and falls below chance in CV).

--channels adds a per-channel analysis: all features of ONE channel (linear probe), and the mean by scalp region
when the loader provides channel names.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from neurotext.audit import clean, fmt, safe_ci, save_json, sfreq_of
from neurotext.experiment import load_data
from neurotext.features import GROUPS, amplitude_normalise, feature_table, region_of, trim_edges
from neurotext.privacy.probes import PROBES, identity_leakage
from neurotext.privacy.verification import verification
from neurotext.stats import paired_diff_ci, verdict


def score(x, who, groups, probes, seed, people_fit, people_eval):
    """Closed-set accuracy for every probe on everybody, and LDA / cosine verification on a held-out half."""
    out = {}
    for p in probes:
        try:
            out[p] = identity_leakage(x, who, groups=groups, n_splits=4, seed=seed, probe=p, standardize=True)
        except ValueError:
            out[p] = None
    ev, fit = np.isin(who, people_eval), np.isin(who, people_fit)
    try:
        v = verification(x[ev], who[ev], groups[ev], x[fit], who[fit], n_pairs=10000, seed=seed)
        out["verify"] = v
    except ValueError:
        out["verify"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--probes", nargs="+", default=list(PROBES), choices=list(PROBES))
    ap.add_argument("--channels", action="store_true", help="per-channel and per-region analysis (linear probe)")
    ap.add_argument("--channel-seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.seeds < 3:
        ap.error("use at least 3 seeds")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg["data"]["source"] == "eegmmidb" and cfg["data"].get("holdout", "exclude") != "exclude":
        ap.error("exploratory analysis must not touch the held-out people: data.holdout must be 'exclude'")

    data = load_data({**cfg["data"], "seed": 0})
    fs = sfreq_of(data, cfg)
    who = np.asarray(data.subject_ids)
    block = cfg["split"].get("block_size") or 1
    groups = np.asarray(data.sentence_ids) // block
    people = np.unique(who)
    chance = 1.0 / len(people)
    print(f"data: {len(data.x)} recordings, {len(people)} people (chance {chance:.3f}), sampling rate {fs:g} Hz\n")

    tables = {"raw": feature_table(data.x, fs)}
    tables["amplitude-normalised"] = feature_table(amplitude_normalise(data.x), fs)
    tables["edges trimmed 10%"] = feature_table(trim_edges(data.x, 0.1), fs)
    tables["normalised + trimmed"] = feature_table(trim_edges(amplitude_normalise(data.x), 0.1), fs)
    base = tables["raw"]
    present = [g for g in GROUPS if g in base.groups]

    settings = {"all features": ("all", "raw", base.columns())}
    for g in present:
        settings[f"only {g}"] = ("only", "raw", base.columns(only=(g,)))
    for g in present:
        settings[f"all minus {g}"] = ("drop", "raw", base.columns(drop=(g,)))
    for name in ("amplitude-normalised", "edges trimmed 10%", "normalised + trimmed"):
        settings[f"control: {name}"] = ("control", name, tables[name].columns())

    probes = tuple(args.probes)
    runs = {k: [] for k in settings}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(people)
        half = len(perm) // 2
        fit_people, eval_people = perm[:half], perm[half:]
        for name, (kind, tab, cols) in settings.items():
            runs[name].append(score(tables[tab].x[:, cols], who, groups, probes, seed, fit_people, eval_people))
        a = runs["all features"][-1][probes[0]]
        print(f"seed {seed}: all features, {probes[0]} probe: accuracy {a['accuracy']:.3f}" if a else f"seed {seed}: n/a", flush=True)

    def excess(name, p):
        return [r[p]["excess_over_chance"] if r[p] else np.nan for r in runs[name]]

    summary = {"chance": chance, "n_people": int(len(people)), "n_seeds": args.seeds, "settings": {}}
    for name, (kind, tab, cols) in settings.items():
        row = {"kind": kind, "n_columns": int(len(cols)), "table": tab}
        for p in probes:
            row[p] = {"accuracy": safe_ci([r[p]["accuracy"] if r[p] else None for r in runs[name]]),
                      "excess": safe_ci(excess(name, p))}
        row["verify"] = {s: safe_ci([((r["verify"] or {}).get(s) or {}).get("eer") for r in runs[name]])
                         for s in ("cosine", "lda")}
        row["share_of_full"], row["loss_vs_full"] = {}, {}
        for p in probes:
            full, here = np.array(excess("all features", p)), np.array(excess(name, p))
            with np.errstate(divide="ignore", invalid="ignore"):
                row["share_of_full"][p] = safe_ci(here / full)
            ok = ~(np.isnan(full) | np.isnan(here))
            if ok.sum() >= 2:
                d = paired_diff_ci(full[ok], here[ok])
                row["loss_vs_full"][p] = {**d, "verdict": verdict(d)}
        summary["settings"][name] = row

    print(f"\n== accuracy [95% CI over seeds]; chance = {chance:.3f} ==")
    print(f"{'setting':32s}{'cols':>6s}" + "".join(f"{p:>26s}" for p in probes) + f"{'LDA EER':>26s}")
    for name, row in summary["settings"].items():
        print(f"{name:32s}{row['n_columns']:6d}" + "".join(f"{fmt(row[p]['accuracy']):>26s}" for p in probes)
              + f"{fmt(row['verify']['lda']):>26s}")
    print("\n== leakage lost when a group is removed (excess of 'all' minus excess of 'all minus X'; > 0 = the group mattered) ==")
    for g in present:
        row = summary["settings"][f"all minus {g}"]
        print(f"  {g:11s}" + "".join(f"  {p}: {fmt(row['loss_vs_full'].get(p), True)}" for p in probes))

    if args.channels:
        names = list(getattr(data, "meta", {}).get("channel_names", []) or [])
        n_ch = data.x.shape[1]
        chan = []
        for c in range(n_ch):
            cols = base.of_channels([c])
            vals = []
            for seed in range(args.channel_seeds):
                r = identity_leakage(base.x[:, cols], who, groups=groups, n_splits=4, seed=seed, probe="linear", standardize=True)
                vals.append(r["excess_over_chance"])
            chan.append({"channel": c, "name": names[c] if c < len(names) else None,
                         "region": region_of(names[c]) if c < len(names) else None, "excess": safe_ci(vals)})
        summary["channels"] = chan
        if names:
            regions = {}
            for row in chan:
                regions.setdefault(row["region"], []).append(row["excess"]["mean"])
            summary["regions"] = {k: {"mean_excess": float(np.mean(v)), "n_channels": len(v)} for k, v in regions.items()}
            print("\n== mean single-channel excess over chance by scalp region (linear probe) ==")
            for k, v in sorted(summary["regions"].items(), key=lambda kv: -kv[1]["mean_excess"]):
                print(f"  {k:10s} {v['mean_excess']:+.3f}  ({v['n_channels']} channels)")
        best = sorted(chan, key=lambda r: -r["excess"]["mean"])[:5]
        print("  strongest single channels:", ", ".join(f"{r['name'] or r['channel']} ({r['excess']['mean']:+.3f})" for r in best))

    save_json(Path(args.out or Path(cfg.get("output_dir", "results")) / "feature_ablation.json"),
              {"config": cfg, "args": vars(args), "summary": summary, "exploratory": True})


if __name__ == "__main__":
    main()
