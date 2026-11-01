"""Classical baseline on ZuCo band-power features, over many seeds, with a sensitivity check.

    python experiments/ridge_baseline.py --config configs/zuco2_nr.yaml --seeds 20

Question: do the precomputed band-power features (8 bands x 105 channels per sentence) carry
ANY information about which sentence was read? Ridge regression maps features to the text
embedding; the right sentence is retrieved among the held-out ones.

For each design (new people / same person) and each planted strength, the table gives the
model, the strongest baseline (noise input, shuffled labels, chance) and the gain over it.
Strength 0 is the real data. Strengths > 0 add a known text-dependent pattern to the real
features: they show how weak a signal this pipeline could still detect, so that a null
result is a bounded statement ("nothing above about X") and not just "nothing found".

Seeds change only the splits here (the data are fixed), so the intervals understate the
true uncertainty: read them as a lower bound.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from neurotext.classical import cross_person, within_person
from neurotext.data import load_zuco2_nr_bandpower
from neurotext.stats import mean_ci, verdict


def fmt(ci):
    return f"{ci['mean']:.3f} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"


def summarize(results: list[dict]) -> dict:
    model = mean_ci([r["scores"]["model"]["top1"] for r in results])
    base = mean_ci([max(v["top1"] for k, v in r["scores"].items() if k != "model") for r in results])
    gain1 = mean_ci([r["gain_top1"] for r in results])
    gain5 = mean_ci([r["gain_top5"] for r in results])
    return {"model_top1": model, "strongest_baseline_top1": base, "gain_top1": {**gain1, "verdict": verdict(gain1)},
            "gain_top5": {**gain5, "verdict": verdict(gain5)}, "n_candidates": results[0]["n_candidates"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--plant", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4],
                        help="planted strengths for the new-people design")
    parser.add_argument("--plant-within", type=float, nargs="+", default=[0.0, 0.05],
                        help="planted strengths for the same-person design (slower)")
    parser.add_argument("--block-size", type=int, default=None,
                        help="hold out contiguous blocks of this many sentences instead of random sentences")
    parser.add_argument("--designs", nargs="+", default=["cross_person", "within_person"],
                        choices=["cross_person", "within_person"])
    args = parser.parse_args()
    if args.seeds < 3:
        parser.error("use at least 3 seeds")
    cfg = yaml.safe_load(Path(args.config).read_text())
    d = cfg["data"]
    if d["source"] != "zuco2_nr":
        parser.error("this baseline reads ZuCo band-power features (source: zuco2_nr)")

    data = load_zuco2_nr_bandpower(
        d["root"], cache_dir=d.get("cache_dir"), min_seconds_per_word=d.get("min_seconds_per_word", 0.05),
        text_dim=d.get("text_dim", 64), progress=print)
    print(f"\n{len(data.x)} recordings, {len(data.subject_codes)} people, {data.x.shape[1]} features each.")
    print("Seeds change only the splits: intervals are a lower bound on the true uncertainty.\n")
    split = cfg["split"]

    out = {"features": data.x.shape[1], "n_recordings": int(len(data.x)), "seeds": args.seeds, "designs": {}}
    out["block_size"] = args.block_size
    how = "random sentences" if args.block_size is None else f"blocks of {args.block_size} consecutive sentences"
    print(f"Held-out sentences: {how}. The baseline 'position_oracle' sees no brain data but is told where in the "
          "session each sentence was read.\n")
    for name, design in (("cross_person", cross_person), ("within_person", within_person)):
        if name not in args.designs:
            continue
        out["designs"][name] = {}
        first = True
        for strength in (args.plant if name == "cross_person" else args.plant_within):
            results = [design(data, seed, split["val_frac"], split["test_frac"], plant=strength, block_size=args.block_size)
                       for seed in range(args.seeds)]
            s = summarize(results)
            s["position_oracle_top1"] = mean_ci([r["scores"]["position_oracle"]["top1"] for r in results])
            out["designs"][name][str(strength)] = s
            if first:
                title = "new people, unseen sentences" if name == "cross_person" else "same person, unseen sentences"
                print(f"== {name} ({title}); chance top1 = {1 / s['n_candidates']:.3f} ({s['n_candidates']} candidates) ==")
                print("  planted | model top1 [95% CI]    | best baseline | oracle position | gain top1 [95% CI]        | verdict")
                first = False
            g = s["gain_top1"]
            print(f"  {strength:6.2f}  | {fmt(s['model_top1'])} | {s['strongest_baseline_top1']['mean']:.3f}         | {s['position_oracle_top1']['mean']:.3f}           | "
                  f"{g['mean']:+.3f} [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}] | {g['verdict']}", flush=True)
        print()

    suffix = "" if args.block_size is None else f"_blocks{args.block_size}"
    path = Path(cfg.get("output_dir", "results")) / f"ridge_baseline{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
