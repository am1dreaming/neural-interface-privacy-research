import numpy as np
import pytest

from neurotext.classical import cross_person, plant_signal, standardize_per_subject, within_person
from neurotext.data import BANDS, EEGData


def synth(signal, n_subj=10, n_sent=60, n_feat=40, seed=0):
    """Every person reads every sentence. Features = person offset + noise + signal * f(text)."""
    rng = np.random.default_rng(seed)
    text = rng.standard_normal((n_sent, 8))
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    w = rng.standard_normal((8, n_feat))
    subj = np.repeat(np.arange(n_subj), n_sent)
    sent = np.tile(np.arange(n_sent), n_subj)
    x = rng.standard_normal((len(subj), n_feat)) + 5 * rng.standard_normal((n_subj, n_feat))[subj]
    x = x + signal * (text[sent] @ w)
    return EEGData(x.astype(np.float32), subj, sent, text.astype(np.float32))


def test_standardize_per_subject_gives_zero_mean_unit_std_for_everyone():
    d = synth(0.0)
    z = standardize_per_subject(d.x, d.subject_ids)
    for s in range(10):
        m = d.subject_ids == s
        assert np.allclose(z[m].mean(0), 0, atol=1e-4) and np.allclose(z[m].std(0), 1, atol=1e-3)


def test_planting_zero_changes_nothing_and_positive_strength_does():
    d = synth(0.0)
    z = standardize_per_subject(d.x, d.subject_ids)
    assert plant_signal(z, d.sentence_ids, d.text_emb, 0.0) is z
    assert not np.allclose(plant_signal(z, d.sentence_ids, d.text_emb, 0.5), z)


def test_cross_person_finds_a_strong_signal():
    r = cross_person(synth(1.5), seed=0)
    chance = 1 / r["n_candidates"]
    assert r["scores"]["model"]["top1"] > 4 * chance and r["gain_top1"] > 0.2


def test_cross_person_finds_nothing_in_pure_noise():
    r = cross_person(synth(0.0), seed=0)
    assert abs(r["gain_top1"]) < 0.15  # chance is 1/15; single seed, so a loose bound
    assert r["scores"]["model"]["top1"] < 0.25


def test_within_person_finds_a_strong_signal_and_nothing_in_noise():
    strong = within_person(synth(1.5), seed=0)
    assert strong["gain_top1"] > 0.2 and strong["n_people"] == 10
    # On pure noise the gain over the strongest of five baselines is slightly NEGATIVE by
    # construction (a maximum of noisy values is above their mean), so the property to check
    # is "no false positive": the mean gain over several seeds must not be positive.
    gains = [within_person(synth(0.0), seed=s)["gain_top1"] for s in range(5)]
    assert np.mean(gains) < 0.05


def test_the_sensitivity_check_detects_planted_signal_in_pure_noise_features():
    """The experiment's logic: real data replaced by noise, a known pattern planted on top."""
    d = synth(0.0)
    assert cross_person(d, seed=1, plant=0.0)["scores"]["model"]["top1"] < 0.25
    assert cross_person(d, seed=1, plant=0.6)["scores"]["model"]["top1"] > 0.4


def test_baselines_are_reported_next_to_the_model():
    r = cross_person(synth(0.0), seed=0)
    assert {"model", "noise_input", "shuffled_labels", "chance"} <= set(r["scores"])


def test_same_seed_same_result():
    a, b = cross_person(synth(0.5), seed=3), cross_person(synth(0.5), seed=3)
    assert a["scores"] == b["scores"] and a["alpha"] == b["alpha"]


# ---- reading the authors' band-power features from a ZuCo-style file ----------------------
h5py = pytest.importorskip("h5py")

from neurotext.data import load_zuco2_nr_bandpower, read_band_features

SENTENCES = [f"Sentence {k} mentions topic{k} and subject{k} clearly." for k in range(8)]


def make_fake_mat(path, lengths, bad_band_at=(), seed=0):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        sd = f.create_group("sentenceData")
        names = ["content", "rawData", "wordbounds"] + [f"mean_{b}" for b in BANDS]
        ds = {k: sd.create_dataset(k, (len(SENTENCES), 1), dtype=h5py.ref_dtype) for k in names}
        for i, s in enumerate(SENTENCES):
            ds["content"][i, 0] = refs.create_dataset(f"c{i}", data=np.array([[ord(c)] for c in s], dtype=np.uint16)).ref
            raw = np.zeros((1, 1)) if lengths[i] is None else rng.standard_normal((lengths[i], 105))
            ds["rawData"][i, 0] = refs.create_dataset(f"r{i}", data=raw).ref
            ds["wordbounds"][i, 0] = refs.create_dataset(f"w{i}", data=np.zeros((4, 10))).ref
            for b in BANDS:
                v = rng.standard_normal((105, 1)) + 1
                if i in bad_band_at and b == "g2":
                    v[3] = np.nan
                ds[f"mean_{b}"][i, 0] = refs.create_dataset(f"{b}{i}", data=v).ref
    return path


def test_band_features_have_840_columns_and_the_same_screen_as_the_raw_loader(tmp_path):
    #          ok    missing  too short  ok    ok    NaN band  ok    ok
    lengths = [1000, None, 40, 2000, 1500, 1200, 900, 1100]
    p = make_fake_mat(tmp_path / "resultsAAA_NR.mat", lengths, bad_band_at={5})
    r = read_band_features(p)
    assert r.features.shape == (8, 8 * 105)
    assert np.isfinite(r.features).all(axis=1).tolist() == [True, False, False, True, True, False, True, True]
    assert r.dropped == {"no_eeg": 1, "too_short": 1, "non_finite": 1}
    # both readers apply the same no_eeg / too_short screen; they differ only in their own
    # NaN checks (here sentence 5 has a NaN in a band feature but finite raw EEG)
    from neurotext.data import read_subject_file
    raw = read_subject_file(p, n_times=64)
    assert raw.sentence_idx.tolist() == [0, 3, 4, 5, 6, 7]
    assert raw.dropped["no_eeg"] == r.dropped["no_eeg"] and raw.dropped["too_short"] == r.dropped["too_short"]


def test_band_feature_loader_gives_the_common_interface_and_uses_the_cache(tmp_path):
    make_fake_mat(tmp_path / "resultsAAA_NR.mat", [1000] * 8, seed=1)
    make_fake_mat(tmp_path / "resultsBBB_NR.mat", [1000, None] + [1000] * 6, seed=2)
    d = load_zuco2_nr_bandpower(tmp_path, cache_dir=tmp_path / "c", text_dim=4)
    assert d.x.shape == (15, 840) and d.subject_codes == ["AAA", "BBB"]
    assert np.bincount(d.subject_ids).tolist() == [8, 7] and d.text_emb.shape[0] == 8
    assert list((tmp_path / "c").glob("*_bands_*.npz"))
    d2 = load_zuco2_nr_bandpower(tmp_path, cache_dir=tmp_path / "c", text_dim=4)
    assert np.array_equal(d.x, d2.x)


# ---- the "is it just drift?" controls ---------------------------------------------------------
from neurotext.classical import position_oracle
from neurotext.data import subject_and_sentence_split


def _truth_for(split, d):
    ts = split.groups["test_sentences"]
    to = {s: i for i, s in enumerate(ts)}
    return ts, np.array([to[s] for s in d.sentence_ids[split.test]])


def test_oracle_position_is_at_chance_when_text_has_no_order_structure():
    tops = []
    for ds in range(4):
        d = synth(0.0, seed=ds)
        for seed in range(15):
            sp = subject_and_sentence_split(d.subject_ids, d.sentence_ids, 0.15, 0.25, seed)
            ts, truth = _truth_for(sp, d)
            tops.append(position_oracle(sp.groups["train_sentences"], sp.groups["val_sentences"], ts, d.text_emb, truth)["top1"])
    assert abs(np.mean(tops) - 1 / 15) < 0.03  # k is picked on validation data, so no optimism


def test_oracle_position_detects_text_that_drifts_along_the_order():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.standard_normal((60, 8)) * 0.3, axis=0)
    walk = (walk / np.linalg.norm(walk, axis=1, keepdims=True)).astype(np.float32)
    d = synth(0.0)
    tops = []
    for seed in range(10):
        sp = subject_and_sentence_split(d.subject_ids, d.sentence_ids, 0.15, 0.25, seed)
        ts, truth = _truth_for(sp, d)
        tops.append(position_oracle(sp.groups["train_sentences"], sp.groups["val_sentences"], ts, walk, truth)["top1"])
    assert np.mean(tops) > 0.4


def drift_world(n_subj=8, n_sent=120, n_feat=30, seed=0):
    """No language at all in the features: they only encode WHEN in the session a sentence was read
    (slow drift), while the text drifts along the order (neighbouring sentences share a topic)."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.standard_normal((n_sent, 8)) * 0.3, axis=0)
    text = (walk / np.linalg.norm(walk, axis=1, keepdims=True)).astype(np.float32)
    t = np.arange(n_sent) / n_sent
    clock = np.stack([f(2 * np.pi * k * t) for k in (1, 2, 3, 4) for f in (np.sin, np.cos)] + [t], axis=1)
    mix = rng.standard_normal((clock.shape[1], n_feat))
    subj = np.repeat(np.arange(n_subj), n_sent)
    sent = np.tile(np.arange(n_sent), n_subj)
    x = 2.0 * clock[sent] @ mix + rng.standard_normal((len(sent), n_feat))
    return EEGData(x.astype(np.float32), subj, sent, text)


def test_drift_fools_random_splits_but_the_controls_expose_it():
    d = drift_world()
    random_split = [within_person(d, seed) for seed in range(6)]
    top1 = np.mean([r["scores"]["model"]["top1"] for r in random_split])
    oracle = np.mean([r["scores"]["position_oracle"]["top1"] for r in random_split])
    chance = 1 / random_split[0]["n_candidates"]
    assert top1 > 2.5 * chance                    # without language, the model still "reads" the sentences
    assert oracle >= top1 - 0.05                  # ...and the oracle-position baseline explains all of it
    assert np.mean([r["gain_top1"] for r in random_split]) < 0.05   # so there is no gain to claim


def test_blocked_splits_remove_most_of_what_drift_gave():
    d = drift_world()
    rand = np.mean([within_person(d, s)["scores"]["model"]["top1"] for s in range(6)])
    block = np.mean([within_person(d, s, block_size=20)["scores"]["model"]["top1"] for s in range(6)])
    assert block < 0.6 * rand
