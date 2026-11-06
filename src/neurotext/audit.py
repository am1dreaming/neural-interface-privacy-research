"""Small helpers shared by the audit experiments (encoders, scaling curve, feature ablation, final run)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .stats import mean_ci


def sfreq_of(data, cfg: dict) -> float:
    """Sampling rate of a dataset: the loader's own, or `features.sfreq` in the config (synthetic data)."""
    fs = getattr(data, "sfreq", None)
    return float(fs or cfg.get("features", {}).get("sfreq", 100.0))


def clean(values) -> list[float]:
    """Drop None / NaN (a population too small to score returns None)."""
    return [float(v) for v in values if v is not None and not np.isnan(v)]


def safe_ci(values) -> dict | None:
    v = clean(values)
    return mean_ci(v) if v else None


def fmt(ci: dict | None, signed: bool = False) -> str:
    if not ci:
        return "n/a"
    f = "+.3f" if signed else ".3f"
    return f"{format(ci['mean'], f)} [{format(ci['ci_low'], f)}, {format(ci['ci_high'], f)}]"


def guard_single_run(path: Path, rerun_because: str | None) -> None:
    """Pre-registered runs are executed once. A repeat needs a reason, which is logged for the
    'Deviations' section of PREREGISTRATION.md."""
    if not path.exists():
        return
    if not rerun_because:
        raise SystemExit(f"{path} already exists. This run is pre-registered as a single run; "
                         f"to repeat it for a technical reason pass --rerun-because '<reason>' "
                         f"and copy the reason into 'Deviations' in PREREGISTRATION.md.")
    log = path.parent / "deviations.log"
    with open(log, "a") as f:
        f.write(f"{path.name}: rerun because: {rerun_because}\n")
    print(f"logged rerun reason in {log}")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"\nsaved {path}")
