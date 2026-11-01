import numpy as np
import pytest

from neurotext.defence import identity_null_defence, noise_defence, pca_defence
from neurotext.privacy import identity_leakage


def embeddings(n_people=6, per=60, dim=16, identity=3.0, seed=0):
    rng = np.random.default_rng(seed)
    who = np.repeat(np.arange(n_people), per)
    z = identity * rng.standard_normal((n_people, dim))[who] + rng.standard_normal((len(who), dim))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z.astype(np.float32), who


def test_noise_defence_keeps_unit_norm_and_grows_with_sigma():
    z, _ = embeddings()
    drift = [np.linalg.norm(noise_defence(z, s, seed=1) - z, axis=1).mean() for s in (0.1, 0.5, 2.0)]
    assert drift[0] < drift[1] < drift[2]
    assert np.allclose(np.linalg.norm(noise_defence(z, 1.0), axis=1), 1, atol=1e-5)


def test_noise_defence_is_reproducible_and_seed_dependent():
    z, _ = embeddings()
    assert np.array_equal(noise_defence(z, 1.0, seed=3), noise_defence(z, 1.0, seed=3))
    assert not np.array_equal(noise_defence(z, 1.0, seed=3), noise_defence(z, 1.0, seed=4))


def test_pca_defence_limits_the_rank_to_k():
    z, _ = embeddings()
    for k in (1, 3, 8):
        assert np.linalg.matrix_rank(pca_defence(z, z, k) - z.mean(0), tol=1e-4) == k


def test_pca_defence_is_fitted_on_the_fit_set_only():
    z, _ = embeddings()
    other, _ = embeddings(seed=5)
    a = pca_defence(z[:100], other, 4)
    b = pca_defence(z[:100], other[:50], 4)
    assert np.allclose(a[:50], b)  # the result for a row does not depend on which other rows are released


def _separation(z, who):
    means = np.stack([z[who == c].mean(0) for c in np.unique(who)])
    return float(np.linalg.norm(means - means.mean(0)))


def test_identity_nulling_removes_the_between_person_subspace_step_by_step():
    z, who = embeddings()                     # 6 people -> the means span 5 dimensions
    sep = [_separation(identity_null_defence(z, who, z, m), who) for m in (0, 1, 3, 5)]
    assert sep[0] > sep[1] > sep[2] > sep[3]
    assert sep[3] < 1e-5                      # all linear differences between the people's means are gone


def test_identity_nulling_does_not_stop_a_nonlinear_attacker_when_it_removes_too_little():
    z, who = embeddings()
    after = identity_leakage(identity_null_defence(z, who, z, m=1), who, probe="mlp", standardize=True)["accuracy"]
    assert after > 0.5  # one direction removed out of many: the person is still recoverable


def test_identity_nulling_never_asks_for_more_directions_than_exist():
    z, who = embeddings(n_people=3)
    assert identity_null_defence(z, who, z, m=50).shape == z.shape


def test_identity_nulling_fitted_on_some_people_does_not_remove_the_identity_of_other_people():
    """The point of measuring on NEW users: directions found on the training people need not
    describe how unseen people differ from each other."""
    z, who = embeddings(n_people=12, seed=0)
    train, new = who < 6, who >= 6
    out = identity_null_defence(z[train], who[train], z, m=5)
    assert _separation(out[train], who[train]) < 1e-5
    assert _separation(out[new], who[new]) > 0.5 * _separation(z[new], who[new])


# ---- training against the adversary -------------------------------------------------------------
torch = pytest.importorskip("torch")

from neurotext.data import make_synthetic
from neurotext.models import EEGEncoder, TextProjector
from neurotext.training import _GradientReversal, fit_alignment_adversarial


def test_gradient_reversal_is_identity_forward_and_flips_the_gradient_backward():
    x = torch.randn(4, 3, requires_grad=True)
    y = _GradientReversal.apply(x, 2.5)
    assert torch.equal(y, x)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -2.5))


def _tiny():
    d = make_synthetic(n_subjects=6, n_sentences=24, trials_per_sentence=1, n_channels=8, n_times=32,
                       text_dim=8, snr=1.5, subject_variability=1.0, seed=0)
    torch.manual_seed(0)
    return d, EEGEncoder(8, 32, d_model=16, out_dim=8, n_layers=1, n_heads=2, patch=16), TextProjector(8, 8)


def test_adversarial_training_runs_reports_both_losses_and_is_deterministic():
    d, enc, proj = _tiny()
    enc2, proj2 = _tiny()[1:]
    kw = dict(adv_weight=2.0, epochs=3, batch_size=32, seed=1)
    a = fit_alignment_adversarial(enc, proj, d.x, d.sentence_ids, d.subject_ids, d.text_emb, **kw)
    b = fit_alignment_adversarial(enc2, proj2, d.x, d.sentence_ids, d.subject_ids, d.text_emb, **kw)
    assert len(a[0]) == len(a[1]) == 3 and a == b


def test_adversary_weight_changes_the_encoder():
    d, enc0, proj0 = _tiny()
    enc1, proj1 = _tiny()[1:]
    kw = dict(epochs=3, batch_size=32, seed=1)
    fit_alignment_adversarial(enc0, proj0, d.x, d.sentence_ids, d.subject_ids, d.text_emb, adv_weight=0.0, **kw)
    fit_alignment_adversarial(enc1, proj1, d.x, d.sentence_ids, d.subject_ids, d.text_emb, adv_weight=8.0, **kw)
    assert any(not torch.equal(a, b) for a, b in zip(enc0.parameters(), enc1.parameters()))


def test_run_defence_seed_end_to_end_on_a_tiny_config():
    from neurotext.defence import run_defence_seed

    cfg = {
        "data": {"source": "synthetic", "n_subjects": 12, "n_sentences": 40, "trials_per_sentence": 1,
                 "n_channels": 8, "n_times": 32, "text_dim": 8, "snr": 1.5, "subject_variability": 1.0},
        "split": {"val_frac": 0.15, "test_frac": 0.25},
        "model": {"d_model": 16, "out_dim": 8, "n_layers": 1, "n_heads": 2, "patch": 16},
        "train": {"epochs": 2, "lr": 0.001, "batch_size": 32, "temperature": 0.1},
        "defence": {"noise": [1.0], "pca": [2], "identity_null": [2], "adversarial": [1.0]},
    }
    a = run_defence_seed(cfg, seed=0, probes=("linear",))
    assert list(a["defences"]) == ["none", "noise s=1.0", "pca k=2", "id-null m=2", "adversarial l=1.0"]
    for v in a["defences"].values():
        assert 0 <= v["utility"]["top1"] <= 1 and v["identity"]["linear"]["chance"] == pytest.approx(1 / a["n_new_users"])
    assert a == run_defence_seed(cfg, seed=0, probes=("linear",))  # same seed, same numbers


# ---- the tradeoff figure -----------------------------------------------------------------------
def _ci(m, w=0.02):
    return {"mean": m, "ci_low": m - w, "ci_high": m + w}


def _summary():
    d = {"none": {"top1": _ci(0.4), "mlp": _ci(0.55), "linear": _ci(0.3)}}
    for fam, vals in (("noise s", (0.5, 1.0, 2.0)), ("pca k", (2, 4, 8)), ("id-null m", (2, 4, 8)), ("adversarial l", (0.5, 2.0))):
        for i, v in enumerate(vals):
            d[f"{fam}={v}"] = {"top1": _ci(0.4 - 0.08 * (i + 1)), "mlp": _ci(0.55 - 0.1 * (i + 1)), "linear": _ci(0.3 - 0.03 * (i + 1))}
    return {"n_seeds": 10, "chance_top1": 0.033, "chance_identity": 0.167, "defences": d}


def test_figure_is_written_for_a_summary(tmp_path):
    pytest.importorskip("matplotlib")
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "plot_defence_tradeoff", pathlib.Path(__file__).resolve().parents[1] / "experiments" / "plot_defence_tradeoff.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "fig.png"
    mod.plot_tradeoff(_summary(), out)
    assert out.exists() and out.stat().st_size > 20_000
    assert mod.setting_value("noise s=0.25") == 0.25 and mod.setting_value("id-null m=16") == 16
