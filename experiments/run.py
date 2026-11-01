"""One experiment, one seed, full printout.

    python experiments/run.py --config configs/smoke_synthetic.yaml

With `source: synthetic` this is a pipeline check, NOT a scientific result. A single run
is also too noisy to compare conditions: use experiments/run_seeds.py for that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from neurotext.experiment import run_once


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    if cfg["data"]["source"] == "synthetic":
        print("NOTE: synthetic data. This run checks the pipeline; it is not a research result.\n")
    result = run_once(cfg, cfg["seed"], log=print)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(args.config).stem}.json"
    path.write_text(json.dumps({"config": cfg, **result}, indent=2))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
