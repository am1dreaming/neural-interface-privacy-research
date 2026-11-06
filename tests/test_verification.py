import numpy as np
import pytest

from neurotext.privacy.verification import eer_auc, make_pairs, verification


def table(identity, n_subj=6, per=40, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    who = np.repeat(np.arange(n_subj), per)
    grp = np.tile(np.arange(per) // 10, n_subj)
    x = identity * rng.standard_normal((n_subj, dim))[who] + rng.standard_normal((len(who), dim))
    return x, who, grp


def test_eer_auc_on_known_cases():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(500, bool), np.zeros(500, bool)]
    perfect = eer_auc(np.r_[rng.normal(2, .1, 500), rng.normal(-2, .1, 500)], y)
    assert perfect == {"eer": 0.0, "auc": 1.0}
    flipped = eer_auc(-np.r_[rng.normal(2, .1, 500), rng.normal(-2, .1, 500)], y)
    assert flipped["auc"] == 0.0 and flipped["eer"] == 1.0
    noise = eer_auc(rng.standard_normal(20000), rng.random(20000) < .5)
    assert abs(noise["eer"] - .5) < .03 and abs(noise["auc"] - .5) < .03
    assert eer_auc(np.zeros(10), np.r_[np.ones(5, bool), np.zeros(5, bool)]) == {"eer": 0.5, "auc": 0.5}


def test_eer_needs_both_kinds_of_pair():
    with pytest.raises(ValueError):
        eer_auc(np.arange(5.0), np.ones(5, bool))


def test_pairs_are_labelled_correctly_and_never_share_a_group():
    _, who, grp = table(1.0)
    i, j, same = make_pairs(who, grp, 1000, seed=3)
    assert (grp[i] != grp[j]).all() and ((who[i] == who[j]) == same).all()
    assert same.sum() == 1000 and (~same).sum() == 1000
    assert all(np.array_equal(a, b) for a, b in zip(make_pairs(who, grp, 50, 1), make_pairs(who, grp, 50, 1)))


def test_cosine_verification_finds_planted_identity_and_ignores_none():
    x, who, grp = table(2.0)
    assert verification(x, who, grp, n_pairs=2000)["cosine"]["eer"] < 0.25
    x0, who, grp = table(0.0)
    assert abs(verification(x0, who, grp, n_pairs=2000)["cosine"]["eer"] - 0.5) < 0.06


def test_lda_score_uses_other_people_and_beats_cosine_when_noise_dominates():
    rng = np.random.default_rng(1)
    dim = 30
    shared = rng.standard_normal((3, dim))          # identity lives in a few directions only
    def people(n, seed):
        r = np.random.default_rng(seed)
        who = np.repeat(np.arange(n), 40)
        x = r.standard_normal((n, 3)) @ shared * 2.0
        return x[who] + 1.5 * r.standard_normal((len(who), dim)), who, np.tile(np.arange(40) // 10, n)
    fit_x, fit_y, _ = people(20, 10)
    x, who, grp = people(6, 11)
    out = verification(x, who, grp, fit_x, fit_y, n_pairs=3000)
    assert out["lda"] is not None and out["lda"]["eer"] < out["cosine"]["eer"]


def test_evaluate_identity_returns_verification_for_new_users_only():
    from neurotext.leakage import evaluate_identity

    x, who, grp = table(2.0, n_subj=8, per=40)
    sent = np.tile(np.arange(40), 8)
    new = np.isin(who, [6, 7])
    out = evaluate_identity({"r": x}, who, sent, new, block_size=10, probes=("linear",), verify=True)
    assert out["new_users"]["r"]["verify"]["cosine"]["eer"] < 0.4
    assert out["new_users"]["r"]["verify"]["lda"] is not None
    assert "verify" not in out["everyone"]["r"]
    plain = evaluate_identity({"r": x}, who, sent, new, block_size=10, probes=("linear",))
    assert "verify" not in plain["new_users"]["r"]
