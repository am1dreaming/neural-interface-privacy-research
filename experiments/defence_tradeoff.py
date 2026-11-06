"""Price of privacy: identity leakage versus decoding accuracy, for several defences.

    python experiments/defence_tradeoff.py --config configs/defence_synthetic.yaml --seeds 10

For every seed the plain encoder and the adversarially trained ones are trained, the embeddings
are released under each defence, and two things are measured for the NEW users (people never seen
in training), on sentences never seen in training:

  utility   top-1 / top-5 retrieval of the right sentence from the defended embedding;
  leakage   accuracy of an adaptive attacker (linear probe and MLP probe) who names the person
            from defended embeddings of other sentences (enrolment).

Each defence is compared with "none" PAIRED BY SEED. Read the verdict columns as:
  leak  "down" = identity accuracy fell significantly, "same" = no evidence of a change
  util  "down" = decoding accuracy fell significantly, "same" = no evidence of a change
A useful defence is "leak down, util same". Intervals are 95% over seeds; on synthetic data seeds
regenerate the data, so they are genuine replicates, but 'new users' are only a handful per seed.

Synthetic data: this measures how the defences behave when a decoder works. It says nothing about
real EEG, where the decoder is at chance and the price would be measured from zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from neurotext.defence import run_defence_seed
from neurotext.privacy.probes import PROBES
from neurotext.stats import mean_ci, paired_diff_ci, verdict


def fmt(ci: dict) -> str:
    return f"{ci['mean']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"


def arrow(ci: dict) -> str:
    v = verdict(ci)
    return {"better": "up", "worse": "down"}.get(v, "same")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--probes", nargs="+", default=list(PROBES), choices=list(PROBES))
    parser.add_argument("--verify", action="store_true", help="also score open-set verification (EER / AUC)")
    args = parser.parse_args()
    if args.seeds < 3:
        parser.error("use at least 3 seeds")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg["data"]["source"] == "synthetic":
        print("NOTE: synthetic data: the decoder works here, so the price of a defence is measurable. "
              "This is not a statement about real EEG.\n")

    runs = []
    for seed in range(args.seeds):
        runs.append(run_defence_seed(cfg, seed, tuple(args.probes), verify=args.verify))
        none = runs[-1]["defences"]["none"]
        print(f"seed {seed:2d} done | no defence: top-1 {none['utility']['top1']:.3f}"
              + "".join(f" | {p} id-acc {none['identity'][p]['accuracy']:.3f}" for p in args.probes if none["identity"][p]),
              flush=True)

    names = list(runs[0]["defences"])
    chance_top1 = float(np.mean([1 / r["n_candidates"] for r in runs]))
    chance_id = float(np.mean([1 / r["n_new_users"] for r in runs]))
    get_top1 = lambda r, n: r["defences"][n]["utility"]["top1"]
    get_id = lambda r, n, p: (r["defences"][n]["identity"][p] or {}).get("accuracy", np.nan)

    summary = {"n_seeds": len(runs), "chance_top1": chance_top1, "chance_identity": chance_id, "defences": {}}
    print(f"\n== mean [95% CI] over {len(runs)} seeds; chance: top-1 {chance_top1:.3f}, identity {chance_id:.3f} ==")
    header = f"{'defence':20s} {'top-1 (new users)':>22s}" + "".join(f"{'id-acc ' + p:>24s}" for p in args.probes)
    if args.verify:
        header += f"{'LDA EER (0.5=chance)':>24s}"
    print(header)
    for n in names:
        row = {"top1": mean_ci([get_top1(r, n) for r in runs])}
        line = f"{n:20s} {fmt(row['top1']):>22s}"
        for p in args.probes:
            row[p] = mean_ci([get_id(r, n, p) for r in runs])
            line += f"{fmt(row[p]):>24s}"
        if args.verify:
            for scorer in ("cosine", "lda"):
                vals = [((r["defences"][n]["identity"].get("verify") or {}).get(scorer) or {}).get("eer", np.nan) for r in runs]
                row[f"eer_{scorer}"] = mean_ci([v for v in vals if not np.isnan(v)]) if not all(np.isnan(vals)) else None
            line += f"{fmt(row['eer_lda']) if row.get('eer_lda') else 'n/a':>24s}"
        summary["defences"][n] = row
        print(line)

    print("\n== paired difference to 'none' (defended minus undefended) ==")
    print(f"{'defence':20s} {'d top-1':>22s} {'util':>6s}" + "".join(f"{'d id-acc ' + p:>26s} {'leak':>5s}" for p in args.probes))
    for n in names:
        if n == "none":
            continue
        d_util = paired_diff_ci([get_top1(r, n) for r in runs], [get_top1(r, "none") for r in runs])
        line = f"{n:20s} {d_util['mean']:+.3f} [{d_util['ci_low']:+.3f}, {d_util['ci_high']:+.3f}] {arrow(d_util):>6s}"
        entry = {"d_top1": {**d_util, "util": arrow(d_util)}}
        for p in args.probes:
            d_leak = paired_diff_ci([get_id(r, n, p) for r in runs], [get_id(r, "none", p) for r in runs])
            lk = {"better": "up", "worse": "down"}.get(verdict(d_leak), "same")
            line += f"  {d_leak['mean']:+.3f} [{d_leak['ci_low']:+.3f}, {d_leak['ci_high']:+.3f}] {lk:>5s}"
            entry[f"d_id_{p}"] = {**d_leak, "leak": lk}
        summary["defences"][n]["paired"] = entry
        print(line)

    out = Path(cfg.get("output_dir", "results"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "defence_tradeoff.json"
    path.write_text(json.dumps({"config": cfg, "summary": summary, "runs": runs}, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
