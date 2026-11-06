"""Identity leakage over many seeds, with controls.

    python experiments/identity_leakage.py --config configs/zuco2_nr.yaml --seeds 10

For every seed: train the encoder (same recipe as experiments/run.py), then measure how well a
probe recovers WHO produced each representation, on held-out blocks of sentences. Compared on
the same recordings: the trained encoder, the same encoder before training (the control: leakage
that needs no learning), the band-power features themselves, and a PCA of the same size as the
embedding. Two probes: linear, and a stronger MLP.

Reading the table: values are accuracy MINUS chance (chance = 1 / number of people), mean and 95%
interval over seeds. The comparison that matters for the encoder is "trained minus untrained",
paired by seed: positive means training added identity information.

Seeds change the encoder's initialisation and which people count as "new users", but the data are
the same, so the intervals are a lower bound on the true uncertainty (new_users is only a few people).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from neurotext.data import EEGData, load_zuco2_nr, load_zuco2_nr_bandpower
from neurotext.leakage import align_by_key, run_identity_seed
from neurotext.privacy.probes import PROBES
from neurotext.stats import mean_ci, paired_diff_ci, verdict

POPULATIONS = {
    "new_users": "NEW USERS: people the encoder never saw in training",
    "everyone": "EVERYONE: all people, including those the encoder was trained on",
}


def cell(values: list[float]) -> str:
    ci = mean_ci(values)
    return f"{ci['mean']:+.3f} [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs (faster, for a quick look)")
    parser.add_argument("--probes", nargs="+", default=list(PROBES), choices=list(PROBES))
    parser.add_argument("--verify", action="store_true",
                        help="also score open-set verification (same person or not): EER and AUC, cosine and LDA")
    args = parser.parse_args()
    if args.seeds < 3:
        parser.error("use at least 3 seeds")

    cfg = yaml.safe_load(Path(args.config).read_text())
    d = cfg["data"]
    if d["source"] != "zuco2_nr":
        parser.error("this experiment needs the ZuCo band-power table (source: zuco2_nr)")
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if not cfg["split"].get("block_size"):
        print("WARNING: split.block_size is not set; recordings close in time will help the probe.\n")

    common = dict(cache_dir=d.get("cache_dir"), min_seconds_per_word=d.get("min_seconds_per_word", 0.05),
                  text_dim=d.get("text_dim", 64))
    raw = load_zuco2_nr(d["root"], decimate_factor=d.get("decimate_factor", 5), n_times=d.get("n_times", 640),
                        progress=print, **common)
    bands = load_zuco2_nr_bandpower(d["root"], **common)
    a, b = align_by_key(raw, bands)
    print(f"\nrecordings: raw EEG {len(raw.x)}, band power {len(bands.x)}, in both {len(a)} "
          f"({len(raw.x) - len(a)} raw recordings without band features are left out)")
    both = EEGData(raw.x[a], raw.subject_ids[a], raw.sentence_ids[a], raw.text_emb,
                   sentences=raw.sentences, subject_codes=raw.subject_codes, sfreq=raw.sfreq)
    band_x = bands.x[b]

    runs = []
    for seed in range(args.seeds):
        r = run_identity_seed(both, band_x, cfg, seed, tuple(args.probes), verify=args.verify)
        runs.append(r)
        t = r["identity"]["new_users"]["trained encoder"]["linear"]
        u = r["identity"]["new_users"]["untrained encoder"]["linear"]
        f = lambda x: "n/a" if x is None else f"{x['excess_over_chance']:+.3f}"
        print(f"seed {seed:2d}: new users, linear probe, excess over chance: trained {f(t)} | untrained {f(u)}", flush=True)

    reps = list(runs[0]["identity"]["new_users"])
    summary: dict = {"n_seeds": len(runs), "probes": args.probes, "populations": {}}
    for pop, title in POPULATIONS.items():
        chance = np.mean([next(v for v in r["identity"][pop][reps[0]].values() if v)["chance"] for r in runs])
        print(f"\n== {title}; chance = {chance:.3f} ==")
        print("   accuracy minus chance, mean [95% CI] over seeds")
        print(f"   {'representation':34s}" + "".join(f"{p:>28s}" for p in args.probes))
        summary["populations"][pop] = {"chance": float(chance), "representations": {}}
        for rep in reps:
            line = f"   {rep:34s}"
            summary["populations"][pop]["representations"][rep] = {}
            for p in args.probes:
                vals = [r["identity"][pop][rep][p]["excess_over_chance"] for r in runs if r["identity"][pop][rep][p]]
                accs = [r["identity"][pop][rep][p]["accuracy"] for r in runs if r["identity"][pop][rep][p]]
                line += f"{cell(vals):>28s}"
                summary["populations"][pop]["representations"][rep][p] = {
                    "excess": mean_ci(vals), "accuracy": mean_ci(accs), "n": len(vals)}
            print(line)
        print("   paired difference, trained encoder minus untrained encoder:")
        for p in args.probes:
            pairs = [(r["identity"][pop]["trained encoder"][p], r["identity"][pop]["untrained encoder"][p]) for r in runs]
            pairs = [(x["excess_over_chance"], y["excess_over_chance"]) for x, y in pairs if x and y]
            diff = paired_diff_ci([x for x, _ in pairs], [y for _, y in pairs])
            summary["populations"][pop].setdefault("trained_minus_untrained", {})[p] = {**diff, "verdict": verdict(diff)}
            print(f"     {p:7s} {diff['mean']:+.3f} [{diff['ci_low']:+.3f}, {diff['ci_high']:+.3f}] -> {verdict(diff)}")

    if args.verify:
        summary["verify"] = {}
        print("\n== open-set verification, NEW USERS: EER and AUC, mean [95% CI] over seeds (0.5 = chance for both) ==")
        print(f"   {'representation':34s}{'cosine EER':>26s}{'LDA EER':>26s}{'LDA AUC':>26s}")
        for rep in reps:
            row = {}
            for sc in ("cosine", "lda"):
                cells = [(r["identity"]["new_users"][rep].get("verify") or {}).get(sc) for r in runs]
                row[sc] = {k: mean_ci([c[k] for c in cells if c]) for k in ("eer", "auc")} if any(cells) else None
            summary["verify"][rep] = row
            f = lambda c: "n/a" if not c else f"{c['mean']:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"
            print(f"   {rep:34s}{f(row['cosine'] and row['cosine']['eer']):>26s}{f(row['lda'] and row['lda']['eer']):>26s}"
                  f"{f(row['lda'] and row['lda']['auc']):>26s}")

    out = Path(cfg.get("output_dir", "results"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "identity_leakage.json"
    path.write_text(json.dumps({"config": cfg, "summary": summary, "runs": runs}, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
