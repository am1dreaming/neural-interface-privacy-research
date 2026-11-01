import numpy as np
import pytest

from neurotext.data import make_synthetic
from neurotext.efficiency import downsample, select_channels
from neurotext.eval import chance_topk, noise_like, run_baselines, topk_retrieval
from neurotext.eval.baselines import signal_gain
from neurotext.privacy import identity_leakage


def test_perfect_similarity_gives_top1_of_one():
    sim = np.eye(10)
    assert topk_retrieval(sim, np.arange(10))["top1"] == 1.0


def test_ties_count_against_the_model():
    sim = np.zeros((6, 6))  # a model that outputs the same score for everything
    assert topk_retrieval(sim, np.arange(6), ks=(1,))["top1"] == 0.0


def test_random_similarity_is_near_chance():
    rng = np.random.default_rng(0)
    sim = rng.standard_normal((4000, 20))
    top1 = topk_retrieval(sim, rng.integers(0, 20, 4000))["top1"]
    assert abs(top1 - 1 / 20) < 0.02


def test_chance_values():
    assert chance_topk(20) == {"top1": 0.05, "top5": 0.25}
    assert chance_topk(3, ks=(1, 5)) == {"top1": pytest.approx(1 / 3)}


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        topk_retrieval(np.zeros((3, 4)), np.zeros(5, dtype=int))


def test_noise_like_matches_channel_statistics():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 4, 64)).astype(np.float32) * np.array([1, 2, 3, 4], dtype=np.float32)[None, :, None]
    n = noise_like(x, np.random.default_rng(1))
    assert n.shape == x.shape
    assert np.allclose(n.std(axis=(0, 2)), x.std(axis=(0, 2)), rtol=0.1)


def test_a_model_that_ignores_the_signal_has_no_gain():
    """The baseline suite must expose a decoder that does not use the recording."""
    rng = np.random.default_rng(0)
    cand = rng.standard_normal((20, 8))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    x = rng.standard_normal((500, 4, 32)).astype(np.float32)
    true_idx = rng.integers(0, 20, 500)

    fixed = rng.standard_normal(8)
    res = run_baselines(lambda a: np.tile(fixed, (len(a), 1)), x, true_idx, cand)
    assert abs(signal_gain(res)) < 0.06


def test_a_model_that_reads_the_signal_has_positive_gain():
    rng = np.random.default_rng(0)
    cand = rng.standard_normal((20, 8))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    true_idx = rng.integers(0, 20, 500)
    # the "recording" literally contains the answer in channel 0, first 8 samples
    x = rng.standard_normal((500, 4, 32)).astype(np.float32)
    x[:, 0, :8] = cand[true_idx]

    res = run_baselines(lambda a: a[:, 0, :8], x, true_idx, cand)
    assert res["model"]["top1"] > 0.9
    assert signal_gain(res) > 0.8
    assert res["noise_input"]["top1"] < 0.2


def test_prior_only_baseline_is_reported_when_given():
    rng = np.random.default_rng(0)
    cand = np.eye(5)
    res = run_baselines(lambda a: a[:, 0, :5], rng.standard_normal((50, 2, 8)), rng.integers(0, 5, 50),
                        cand, prior_logprob=np.log(np.full(5, 0.2)))
    assert "prior_only" in res


def test_identity_probe_finds_planted_identity():
    rng = np.random.default_rng(0)
    subj = np.repeat(np.arange(5), 60)
    emb = rng.standard_normal((300, 16)) + 3 * np.eye(5)[subj] @ rng.standard_normal((5, 16))
    r = identity_leakage(emb, subj)
    assert r["accuracy"] > 0.9 and r["excess_over_chance"] > 0.5


def test_identity_probe_is_near_chance_on_noise():
    rng = np.random.default_rng(0)
    subj = np.repeat(np.arange(5), 60)
    r = identity_leakage(rng.standard_normal((300, 16)), subj)
    assert abs(r["excess_over_chance"]) < 0.15


def test_grouped_probe_is_near_chance_when_embeddings_carry_only_the_sentence():
    """Embeddings that encode the text and nothing about the person must not 'leak'.
    This is the repeated-measures case that biased the naive random-split probe."""
    rng = np.random.default_rng(0)
    n_subj, n_sent, reps = 5, 24, 2
    subj = np.repeat(np.arange(n_subj), n_sent * reps)
    sent = np.tile(np.repeat(np.arange(n_sent), reps), n_subj)
    sentence_vectors = rng.standard_normal((n_sent, 16))
    emb = sentence_vectors[sent] + 0.05 * rng.standard_normal((len(sent), 16))
    r = identity_leakage(emb, subj, groups=sent)
    assert abs(r["excess_over_chance"]) < 0.12


def test_grouped_probe_still_finds_planted_identity():
    rng = np.random.default_rng(0)
    n_subj, n_sent, reps = 5, 24, 2
    subj = np.repeat(np.arange(n_subj), n_sent * reps)
    sent = np.tile(np.repeat(np.arange(n_sent), reps), n_subj)
    emb = rng.standard_normal((len(sent), 16)) + 3 * rng.standard_normal((n_subj, 16))[subj]
    r = identity_leakage(emb, subj, groups=sent)
    assert r["accuracy"] > 0.9


def test_identity_probe_needs_two_subjects():
    with pytest.raises(ValueError):
        identity_leakage(np.zeros((10, 3)), np.zeros(10, dtype=int))


def test_synthetic_shapes_and_determinism():
    a = make_synthetic(n_subjects=3, n_sentences=5, trials_per_sentence=2, n_channels=4, n_times=32, seed=1)
    b = make_synthetic(n_subjects=3, n_sentences=5, trials_per_sentence=2, n_channels=4, n_times=32, seed=1)
    assert a.x.shape == (30, 4, 32) and a.text_emb.shape[0] == 5
    assert np.array_equal(a.x, b.x)
    assert np.allclose(np.linalg.norm(a.text_emb, axis=1), 1, atol=1e-5)


def test_channel_selection_and_downsampling_shapes():
    x = np.random.default_rng(0).standard_normal((3, 8, 64)).astype(np.float32)
    assert select_channels(x, [0, 3, 5]).shape == (3, 3, 64)
    assert downsample(x, 2).shape == (3, 8, 32)
    assert downsample(x, 1) is x
