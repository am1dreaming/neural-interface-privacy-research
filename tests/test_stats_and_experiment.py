import numpy as np
import pytest

from neurotext.stats import mean_ci, paired_diff_ci, verdict


def test_mean_ci_matches_hand_computation():
    r = mean_ci([1.0, 2.0, 3.0, 4.0, 5.0], confidence=0.95)
    # std = 1.5811, sem = 0.7071, t(0.975, 4) = 2.7764 -> half-width 1.963
    assert r["mean"] == pytest.approx(3.0)
    assert r["ci_high"] - r["mean"] == pytest.approx(1.963, abs=0.005)
    assert r["n"] == 5


def test_ci_covers_the_true_mean_about_95_percent_of_the_time():
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(1000):
        ci = mean_ci(rng.normal(0.3, 1.0, size=10))
        hits += ci["ci_low"] <= 0.3 <= ci["ci_high"]
    assert 0.92 < hits / 1000 < 0.98


def test_single_value_has_no_interval():
    r = mean_ci([0.5])
    assert r["mean"] == 0.5 and np.isnan(r["ci_low"])


def test_paired_difference_uses_pairs_not_marginals():
    # large between-seed variation, but a perfectly consistent +0.05 improvement
    base = np.array([0.10, 0.40, 0.25, 0.60, 0.33])
    better = base + 0.05
    unpaired = mean_ci(better)
    paired = paired_diff_ci(better, base)
    assert paired["ci_low"] > 0 and verdict(paired) == "better"
    # the between-seed spread dominates the unpaired view; pairing removes it
    assert (paired["ci_high"] - paired["ci_low"]) < 0.1 * (unpaired["ci_high"] - unpaired["ci_low"])


def test_verdicts():
    assert verdict({"ci_low": 0.01, "ci_high": 0.1}) == "better"
    assert verdict({"ci_low": -0.1, "ci_high": -0.01}) == "worse"
    assert verdict({"ci_low": -0.05, "ci_high": 0.05}) == "no evidence of a difference"
    assert verdict({"ci_low": float("nan"), "ci_high": float("nan")}) == "not enough seeds"
    assert verdict({"ci_low": 0.01, "ci_high": 0.1}, higher_is_better=False) == "worse"


def test_paired_requires_equal_length():
    with pytest.raises(ValueError):
        paired_diff_ci([1, 2, 3], [1, 2])


def test_run_once_end_to_end_on_a_tiny_config():
    pytest.importorskip("torch")
    from neurotext.experiment import run_once

    cfg = {
        "data": {"source": "synthetic", "n_subjects": 6, "n_sentences": 16, "trials_per_sentence": 2,
                 "n_channels": 8, "n_times": 32, "text_dim": 8, "snr": 1.0, "subject_variability": 0.2},
        "split": {"val_frac": 0.15, "test_frac": 0.25},
        "model": {"d_model": 16, "out_dim": 8, "n_layers": 1, "n_heads": 2, "patch": 16},
        "train": {"epochs": 2, "lr": 0.001, "batch_size": 16, "temperature": 0.1},
        "personalization": {"rank": 2, "epochs": 2, "lr": 0.01, "calibration_sentences": [0, 2]},
    }
    r = run_once(cfg, seed=0, include_efficiency=False)
    assert r["synthetic"] is True
    assert set(r["calibration_curve_top1"]) == {0, 2}
    assert {"model", "noise_input", "shuffled_labels", "chance", "position_oracle"} <= set(r["zero_shot"])
    assert r["n_candidates"] == 4
    # same seed -> same result (reproducibility is a project requirement)
    assert run_once(cfg, seed=0, include_efficiency=False)["zero_shot"] == r["zero_shot"]
