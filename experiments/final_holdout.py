"""The pre-registered final evaluation on the held-out people. Run ONCE.

    python experiments/final_holdout.py --config configs/eegmmidb_final.yaml --seeds 10

The encoder is trained on the development people only (never on the held-out ones); the held-out people are
the NEW people (`split.test_subjects`). Reported against chance 1/n_heldout (n = 22 usable: see PREREGISTRATION.md):

  H1 (a, b)  closed-set accuracy of the linear and MLP probe above chance;
  H1 (c, d)  EER of the cosine and LDA verification scores below 0.5;
  H3 (a)     defence 'id-null m=32': linear-probe accuracy falls compared with no defence;
  H3 (b)     under that defence the MLP probe is still above chance.

One-sided t-tests over seeds, Holm over these 6 tests (PREREGISTRATION.md, family A). The script refuses to
overwrite its output; a technical rerun needs --rerun-because, which is logged in results/deviations.log.
Seeds change initialisation and fold assignment, not the people: the intervals are a lower bound.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from neurotext.audit import clean, fmt, guard_single_run, safe_ci, save_json
from neurotext.data import holdout_ids
from neurotext.defence import run_defence_seed
from neurotext.experiment import load_data
from neurotext.privacy.probes import PROBES
from neurotext.stats import holm, mean_ci, one_sided_p, paired_diff_ci

NULL_DEFENCE = "id-null m=32"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rerun-because", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg["data"]["source"] != "eegmmidb" or cfg["data"].get("holdout") != "include":
        ap.error("the final run needs data.source: eegmmidb and data.holdout: include (configs/eegmmidb_final.yaml)")
    if args.seeds != 10:
        print("NOTE: the pre-registration fixes 10 seeds; a different number is a deviation to be written down.")
    out_path = Path(args.out or Path(cfg.get("output_dir", "results")) / "final_holdout.json")
    guard_single_run(out_path, args.rerun_because)

    data = load_data({**cfg["data"], "seed": 0})
    held = holdout_ids(data)
    if len(held) < 2:
        raise SystemExit("fewer than 2 held-out people in the data")
    cfg["split"]["test_subjects"] = [int(i) for i in held]
    codes = [data.subject_codes[i] for i in held]
    print(f"held-out people used as NEW people: {len(held)} ({', '.join(codes)})")
    print(f"development people (train/val): {len(np.unique(data.subject_ids)) - len(held)}\n")

    runs = []
    for seed in range(args.seeds):
        r = run_defence_seed(cfg, seed, PROBES, verify=True)
        runs.append(r)
        n = r["defences"]["none"]["identity"]["linear"]
        print(f"seed {seed}: no defence, linear probe accuracy {n['accuracy']:.3f} (chance {n['chance']:.3f})", flush=True)

    chance = 1.0 / len(held)
    ident = lambda r, name, p: r["defences"][name]["identity"][p]
    ver = lambda r, name, s: ((r["defences"][name]["identity"].get("verify") or {}).get(s) or {}).get("eer")
    tests, est = {}, {}

    for p in PROBES:
        v = clean([ident(r, "none", p)["accuracy"] if ident(r, "none", p) else None for r in runs])
        tests[f"H1 {p} accuracy > 1/n"] = one_sided_p(v, chance, "greater")
        est[f"H1 {p} accuracy > 1/n"] = {"estimate": mean_ci(v), "null": chance}
    for s in ("cosine", "lda"):
        v = clean([ver(r, "none", s) for r in runs])
        tests[f"H1 {s} EER < 0.5"] = one_sided_p(v, 0.5, "less")
        est[f"H1 {s} EER < 0.5"] = {"estimate": mean_ci(v) if v else None, "null": 0.5}
    if NULL_DEFENCE in runs[0]["defences"]:
        a = [ident(r, NULL_DEFENCE, "linear")["accuracy"] - ident(r, "none", "linear")["accuracy"] for r in runs]
        tests["H3a linear: id-null minus none < 0"] = one_sided_p(a, 0.0, "less")
        est["H3a linear: id-null minus none < 0"] = {"estimate": mean_ci(a), "null": 0.0}
        b = clean([ident(r, NULL_DEFENCE, "mlp")["accuracy"] if ident(r, NULL_DEFENCE, "mlp") else None for r in runs])
        tests["H3b mlp under id-null > 1/n"] = one_sided_p(b, chance, "greater")
        est["H3b mlp under id-null > 1/n"] = {"estimate": mean_ci(b), "null": chance}
    adj = holm(tests)

    print(f"\n== confirmatory family A: one-sided t-tests over {len(runs)} seeds, Holm-adjusted ==")
    print(f"{'test':42s}{'estimate [95% CI]':>28s}{'null':>8s}{'p':>9s}{'Holm p':>9s}  verdict")
    summary = {"n_heldout": len(held), "heldout_codes": codes, "chance": chance, "tests": {}}
    for k in tests:
        ok = bool(adj[k] < 0.05) and (est[k]["estimate"] is not None)
        summary["tests"][k] = {**est[k], "p": tests[k], "p_holm": adj[k], "confirmed": ok}
        print(f"{k:42s}{fmt(est[k]['estimate']):>28s}{est[k]['null']:8.3f}{tests[k]:9.4f}{adj[k]:9.4f}  "
              f"{('CONFIRMED' if ok else 'NOT confirmed') if args.seeds == 10 else ('passes' if ok else 'fails') + ' (seeds != 10: a deviation, not a confirmation)'}")

    print("\n== all defences, new (held-out) people: accuracy, EER, and paired change against no defence ==")
    summary["defences"] = {}
    for name in runs[0]["defences"]:
        row = {}
        for p in PROBES:
            row[p] = safe_ci([ident(r, name, p)["accuracy"] if ident(r, name, p) else None for r in runs])
            if name != "none":
                d = [ident(r, name, p)["accuracy"] - ident(r, "none", p)["accuracy"] for r in runs
                     if ident(r, name, p) and ident(r, "none", p)]
                row[f"d_{p}"] = paired_diff_ci(d, [0.0] * len(d)) if len(d) >= 2 else None
        row["lda_eer"] = safe_ci([ver(r, name, "lda") for r in runs])
        summary["defences"][name] = row
        print(f"  {name:20s}" + "".join(f"  {p}: {fmt(row[p])}" for p in PROBES) + f"  LDA EER: {fmt(row['lda_eer'])}")
    print("\nExploratory (not in the pre-registered family): everything in this last table except the tests above.")
    save_json(out_path, {"config": cfg, "args": vars(args), "summary": summary, "runs": runs})


if __name__ == "__main__":
    main()
