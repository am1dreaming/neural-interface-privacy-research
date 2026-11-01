"""Repeat one experiment over many seeds and report means with confidence intervals.

    python experiments/run_seeds.py --config configs/smoke_synthetic.yaml --seeds 10

Single runs are too noisy to compare conditions (tiny test sets move top-1 by several
points). Here every comparison that matters is *paired by seed*: e.g. calibrated minus
zero-shot accuracy, per seed, with a confidence interval on the difference.

With `source: synthetic` each seed regenerates the data, so seeds are independent
replicates. On real data they are not (see neurotext/stats.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from neurotext.experiment import run_once
from neurotext.stats import mean_ci, paired_diff_ci, verdict


def fmt(ci: dict) -> str:
    return f"{ci['mean']:.3f}  [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    if args.seeds < 3:
        parser.error("use at least 3 seeds for an interval (10 or more is better)")

    cfg = yaml.safe_load(Path(args.config).read_text())
    synthetic = cfg["data"]["source"] == "synthetic"
    if synthetic:
        print("NOTE: synthetic data. This run checks the pipeline; it is not a research result.\n")
    else:
        print("NOTE: real data. Seeds change splits and initialisation but not the data, so the intervals "
              "below understate the true uncertainty: treat them as a lower bound.\n")

    runs = []
    for i, seed in enumerate(range(args.start_seed, args.start_seed + args.seeds)):
        r = run_once(cfg, seed, include_efficiency=(i == 0))
        runs.append(r)
        print(f"seed {seed:3d}: model top1 {r['zero_shot']['model']['top1']:.3f}  "
              f"gain {r['signal_gain_top1']:+.3f}  "
              f"leak excess {r['identity_leakage']['excess_over_chance']:+.3f}", flush=True)

    c = args.confidence
    pct = int(round(c * 100))
    summary: dict = {"n_seeds": len(runs), "confidence": c}
    print(f"\n==== {len(runs)} seeds, mean [{pct}% CI] ====")

    print("\n-- new people, unseen sentences, zero calibration --")
    summary["zero_shot"] = {}
    names = ["model", "noise_input", "shuffled_labels", "chance"]
    if "position_oracle" in runs[0]["zero_shot"]:
        names.append("position_oracle")
    for name in names:
        summary["zero_shot"][name] = {}
        for metric in ("top1", "top5"):
            ci = mean_ci([r["zero_shot"][name][metric] for r in runs], c)
            summary["zero_shot"][name][metric] = ci
        print(f"  {name:16s} top1 {fmt(summary['zero_shot'][name]['top1'])}   top5 {fmt(summary['zero_shot'][name]['top5'])}")
    gain = mean_ci([r["signal_gain_top1"] for r in runs], c)
    summary["signal_gain_top1"] = {**gain, "verdict": verdict(gain)}
    print(f"  signal gain over strongest baseline (top1): {fmt(gain)}  -> {verdict(gain)}")

    print("\n-- calibration (top1 per new person, paired with zero calibration by seed) --")
    cal_keys = sorted(runs[0]["calibration_curve_top1"])
    summary["calibration"] = {}
    for n_cal in cal_keys:
        vals = [r["calibration_curve_top1"][n_cal] for r in runs]
        entry = {"top1": mean_ci(vals, c)}
        line = f"  {n_cal:2d} sentences: top1 {fmt(entry['top1'])}"
        if n_cal != 0 and 0 in cal_keys:
            base = [r["calibration_curve_top1"][0] for r in runs]
            d = paired_diff_ci(vals, base, c)
            entry["diff_vs_zero_calibration"] = {**d, "verdict": verdict(d)}
            line += f"   diff vs 0: {d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] -> {verdict(d)}"
        summary["calibration"][n_cal] = entry
        print(line)

    print("\n-- privacy: linear probe excess accuracy over chance (lower is better) --")
    leak = mean_ci([r["identity_leakage"]["excess_over_chance"] for r in runs], c)
    leak_verdict = "leakage detected" if leak["ci_low"] > 0 else "no evidence of leakage"
    summary["identity_leakage_excess"] = {**leak, "verdict": leak_verdict}
    print(f"  excess {fmt(leak)}  -> {leak_verdict}")

    eff = runs[0]["efficiency"]
    summary["efficiency"] = eff
    print("\n-- efficiency (first seed) --")
    print(f"  {eff}")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(args.config).stem}_seeds.json"
    path.write_text(json.dumps({"config": cfg, "synthetic": synthetic, "summary": summary, "runs": runs}, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
