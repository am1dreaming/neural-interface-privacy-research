"""Small-sample statistics for comparing runs over seeds.

Seeds are only independent replicates when each seed regenerates the *data* too (as
the synthetic source does). On a fixed real dataset, seeds change splits and
initialisation only, so intervals understate the true uncertainty: treat them as a floor.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_ci(values, confidence: float = 0.95) -> dict[str, float]:
    """Mean with a Student-t confidence interval."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n == 0:
        raise ValueError("no values")
    mean = float(v.mean())
    if n < 2:
        return {"mean": mean, "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": n}
    std = float(v.std(ddof=1))
    half = float(stats.t.ppf((1 + confidence) / 2, n - 1) * std / np.sqrt(n))
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half, "n": n}


def paired_diff_ci(a, b, confidence: float = 0.95) -> dict[str, float]:
    """CI of mean(a - b) for paired measurements (same seed, two conditions)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same length")
    return mean_ci(a - b, confidence)


def verdict(ci: dict[str, float], higher_is_better: bool = True) -> str:
    """Plain-language reading of a difference CI."""
    lo, hi = ci["ci_low"], ci["ci_high"]
    if np.isnan(lo):
        return "not enough seeds"
    if lo > 0:
        return "better" if higher_is_better else "worse"
    if hi < 0:
        return "worse" if higher_is_better else "better"
    return "no evidence of a difference"


def one_sided_p(values, null: float = 0.0, alternative: str = "greater") -> float:
    """One-sided t-test p-value that the mean of `values` differs from `null` in the stated direction
    ("greater" or "less"). Seeds are the replicates, so this is optimistic on a fixed real dataset
    (see the module note); report it together with the interval, not instead of it."""
    v = np.asarray(values, dtype=float) - null
    if len(v) < 2:
        return float("nan")
    if np.allclose(v, v[0]):  # zero spread: the t statistic is undefined
        return 0.0 if (v[0] > 0) == (alternative == "greater") and v[0] != 0 else 1.0
    return float(stats.ttest_1samp(v, 0.0, alternative=alternative).pvalue)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values for a family of tests (NaN stays NaN and is not counted)."""
    items = [(k, p) for k, p in pvalues.items() if not np.isnan(p)]
    items.sort(key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for rank, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adj[k] = running
    return {k: adj.get(k, float("nan")) for k in pvalues}
