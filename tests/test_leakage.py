import numpy as np
import pytest

from neurotext.data import EEGData
from neurotext.leakage import _pca, align_by_key, evaluate_identity
from neurotext.privacy import identity_leakage


def people_table(identity, n_subj=8, n_sent=40, dim=12, seed=0):
    """Each person reads every sentence. Rows = person offset (strength `identity`) + sentence
    pattern + noise."""
    rng = np.random.default_rng(seed)
    subj = np.repeat(np.arange(n_subj), n_sent)
    sent = np.tile(np.arange(n_sent), n_subj)
    who = rng.standard_normal((n_subj, dim))
    what = rng.standard_normal((n_sent, dim))
    x = identity * who[subj] + what[sent] + 0.5 * rng.standard_normal((len(subj), dim))
    return x.astype(np.float32), subj, sent


@pytest.mark.parametrize("probe", ["linear", "mlp"])
def test_both_probes_find_planted_identity_and_ignore_pure_text(probe):
    x, subj, sent = people_table(identity=2.0)
    r = identity_leakage(x, subj, groups=sent // 10, probe=probe, standardize=True)
    assert r["accuracy"] > 0.9
    x0, subj, sent = people_table(identity=0.0)
    r0 = identity_leakage(x0, subj, groups=sent // 10, probe=probe, standardize=True)
    assert abs(r0["excess_over_chance"]) < 0.12


def test_unknown_probe_is_rejected():
    x, subj, sent = people_table(1.0)
    with pytest.raises(ValueError, match="unknown probe"):
        identity_leakage(x, subj, probe="oracle")


def test_default_probe_call_is_unchanged_by_the_new_options():
    x, subj, sent = people_table(1.0)
    a = identity_leakage(x, subj, groups=sent // 10)
    b = identity_leakage(x, subj, groups=sent // 10, probe="linear", standardize=False)
    assert a == b


def test_standardize_helps_when_features_have_wildly_different_scales():
    x, subj, sent = people_table(2.0)
    x = x * np.logspace(-3, 3, x.shape[1]).astype(np.float32)
    plain = identity_leakage(x, subj, groups=sent // 10)["accuracy"]
    scaled = identity_leakage(x, subj, groups=sent // 10, standardize=True)["accuracy"]
    assert scaled > 0.9 and scaled >= plain


def test_normalising_by_the_person_inside_cross_validation_is_an_artefact_not_privacy():
    """Why that control was dropped: z-scoring each person over ALL their recordings makes the
    train and test parts mirror images, so the probe is wrong on purpose (accuracy below chance)."""
    x, subj, sent = people_table(identity=3.0)
    z = np.empty_like(x)
    for s_ in np.unique(subj):
        m = subj == s_
        z[m] = (x[m] - x[m].mean(0)) / x[m].std(0)
    r = identity_leakage(z, subj, groups=sent // 10, standardize=True)
    assert r["excess_over_chance"] < 0  # below chance although the data were built to be identifiable


def test_pca_returns_the_requested_size():
    x, _, _ = people_table(1.0, dim=30)
    assert _pca(x, 8).shape == (len(x), 8)


def test_evaluate_identity_reports_both_populations_representations_and_probes():
    x_id, subj, sent = people_table(2.0)
    x_text, _, _ = people_table(0.0)
    new = np.isin(subj, [0, 1, 2])
    res = evaluate_identity({"has identity": x_id, "text only": x_text}, subj, sent, new, block_size=10)
    assert set(res) == {"new_users", "everyone"}
    assert set(res["new_users"]) == {"has identity", "text only"}
    for pop in res.values():
        assert pop["has identity"]["linear"]["accuracy"] > 0.9
        assert abs(pop["text only"]["linear"]["excess_over_chance"]) < 0.15
        assert set(pop["has identity"]) == {"linear", "mlp"}
    assert res["new_users"]["has identity"]["linear"]["chance"] == pytest.approx(1 / 3)
    assert res["everyone"]["has identity"]["linear"]["chance"] == pytest.approx(1 / 8)


def _table(rows, codes, text_dim=4):
    subj = np.array([c for c, _ in rows])
    sent = np.array([t for _, t in rows])
    return EEGData(np.zeros((len(rows), 2), np.float32), subj, sent, np.zeros((50, text_dim), np.float32),
                   subject_codes=codes)


def test_align_by_key_matches_person_codes_not_numeric_ids():
    # the two loaders may number people differently; only (code, sentence) identifies a recording
    raw = _table([(0, 5), (0, 6), (1, 5), (1, 7)], ["AAA", "BBB"])
    # ids swapped (0 = BBB here), rows reordered, and (BBB, 7) is missing
    bands = _table([(0, 5), (1, 6), (1, 5)], ["BBB", "AAA"])
    a, b = align_by_key(raw, bands)
    assert a.tolist() == [0, 1, 2]  # raw rows (AAA,5) (AAA,6) (BBB,5); (BBB,7) has no counterpart
    assert len(b) == 3
    for i, j in zip(a, b):
        assert (raw.subject_codes[raw.subject_ids[i]], raw.sentence_ids[i]) == \
               (bands.subject_codes[bands.subject_ids[j]], bands.sentence_ids[j])


def test_run_identity_seed_end_to_end_on_a_tiny_table():
    pytest.importorskip("torch")
    from neurotext.data import make_synthetic
    from neurotext.leakage import run_identity_seed

    d = make_synthetic(n_subjects=8, n_sentences=32, trials_per_sentence=1, n_channels=8, n_times=32,
                       text_dim=8, snr=1.0, subject_variability=1.0, seed=0)
    bands = d.x.mean(axis=2)  # stand-in for the band-power table, row-aligned with d.x
    cfg = {"split": {"val_frac": 0.15, "test_frac": 0.25, "block_size": 8},
           "model": {"d_model": 16, "out_dim": 8, "n_layers": 1, "n_heads": 2, "patch": 16},
           "train": {"epochs": 2, "lr": 0.001, "batch_size": 32, "temperature": 0.1}}
    a = run_identity_seed(d, bands, cfg, seed=0, probes=("linear",))
    assert a["n_new_users"] == 2
    reps = set(a["identity"]["everyone"])
    assert {"trained encoder", "untrained encoder", "band power (all features)"} <= reps
    assert a["identity"]["everyone"]["trained encoder"]["linear"]["chance"] == pytest.approx(1 / 8)
    b = run_identity_seed(d, bands, cfg, seed=0, probes=("linear",))
    assert a["identity"] == b["identity"]  # same seed, same numbers
