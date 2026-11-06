import json

import pytest

pytest.importorskip("matplotlib")

from neurotext.plots import make_all  # noqa: E402


def ci(m, lo, hi):
    return {"mean": m, "std": 0.01, "ci_low": lo, "ci_high": hi, "n": 5}


def test_every_figure_is_drawn_from_minimal_result_files(tmp_path):
    reps = {"trained encoder": {p: {"accuracy": ci(0.5, 0.4, 0.6)} for p in ("linear", "mlp")},
            "band power (all features)": {p: {"accuracy": ci(0.99, 0.98, 1.0)} for p in ("linear", "mlp")}}
    (tmp_path / "identity_leakage.json").write_text(json.dumps(
        {"summary": {"populations": {"new_users": {"chance": 0.25, "representations": reps},
                                      "everyone": {"chance": 0.0625, "representations": reps}}}}))
    ns = [5, 10]
    cell = lambda n: {"accuracy": ci(0.8, 0.7, 0.9), "excess": ci(0.6, 0.5, 0.7), "chance": 1 / n}
    ver = lambda n: {"eer": ci(0.2, 0.1, 0.3), "auc": ci(0.9, 0.8, 0.95)}
    rep = {"linear": {str(n): cell(n) for n in ns}, "mlp": {str(n): cell(n) for n in ns}}
    vrep = {"lda": {str(n): ver(n) for n in ns}, "cosine": {str(n): ver(n) for n in ns}}
    (tmp_path / "scaling_curve.json").write_text(json.dumps(
        {"summary": {"n_list": ns, "reps": {"trained encoder": rep}, "verify": {"trained encoder": vrep}}}))
    (tmp_path / "defence_tradeoff.json").write_text(json.dumps({"summary": {
        "chance_top1": 0.03, "chance_identity": 0.17,
        "defences": {"none": {"top1": ci(0.37, 0.3, 0.4), "linear": ci(0.3, 0.2, 0.4), "mlp": ci(0.5, 0.4, 0.6), "eer_lda": ci(0.2, 0.1, 0.3)},
                     "pca k=8": {"top1": ci(0.34, 0.3, 0.4), "linear": ci(0.2, 0.1, 0.3), "mlp": ci(0.37, 0.3, 0.4), "eer_lda": ci(0.35, 0.3, 0.4)}}}}))
    row = lambda: {"linear": {"accuracy": ci(0.9, 0.8, 1.0)}, "mlp": {"accuracy": ci(0.95, 0.9, 1.0)}}
    settings = {"all features": row()}
    for g in ("rms", "alpha"):
        settings[f"only {g}"] = row()
        settings[f"all minus {g}"] = row()
    for c in ("amplitude-normalised", "edges trimmed 10%", "normalised + trimmed"):
        settings[f"control: {c}"] = row()
    (tmp_path / "feature_ablation.json").write_text(json.dumps({"summary": {"chance": 0.0625, "settings": settings}}))
    enc = {}
    for name in ("transformer", "linear"):
        for dim in (8, 32):
            enc[f"{name}-{dim}"] = {"encoder": name, "out_dim": dim,
                                    "trained": {p: {"accuracy": ci(0.4, 0.3, 0.5), "chance": 0.25} for p in ("linear", "mlp")},
                                    "trained_minus_untrained": {"linear": {**ci(-0.04, -0.08, 0.0), "verdict": "worse"}},
                                    "verify_trained": {"lda": {"eer": ci(0.3, 0.2, 0.4)}}}
    (tmp_path / "encoder_robustness.json").write_text(json.dumps({"summary": {"encoders": enc, "probes": ["linear", "mlp"]}}))
    made = make_all(tmp_path)
    assert len(made) == 5 and all(p.stat().st_size > 5000 for p in made)
    assert "transformer" in (tmp_path / "encoders_table.md").read_text()


def test_missing_results_are_skipped_not_fatal(tmp_path):
    assert make_all(tmp_path) == []
