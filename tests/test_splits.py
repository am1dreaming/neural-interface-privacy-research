import numpy as np
import pytest

from neurotext.data import (
    LeakageError,
    Split,
    check_no_leakage,
    select_calibration,
    sentence_split,
    subject_and_sentence_split,
    subject_split,
)


@pytest.fixture
def ids():
    subjects = np.repeat(np.arange(12), 40)
    sentences = np.tile(np.repeat(np.arange(20), 2), 12)
    return subjects, sentences


def test_subject_split_has_no_subject_overlap(ids):
    subj, sent = ids
    s = subject_split(subj, seed=1)
    check_no_leakage(s, subj, sent, by=("subject",))


def test_sentence_split_has_no_sentence_overlap(ids):
    subj, sent = ids
    s = sentence_split(sent, seed=1)
    check_no_leakage(s, subj, sent, by=("sentence",))


@pytest.mark.parametrize("seed", range(5))
def test_strict_split_has_no_leakage_of_either_kind(ids, seed):
    subj, sent = ids
    s = subject_and_sentence_split(subj, sent, seed=seed)
    check_no_leakage(s, subj, sent)
    assert len(s.train) and len(s.test)


def test_strict_split_drops_mixed_samples(ids):
    subj, sent = ids
    s = subject_and_sentence_split(subj, sent, seed=0)
    assert len(s.train) + len(s.val) + len(s.test) < len(subj)


def test_splits_are_reproducible(ids):
    subj, sent = ids
    a = subject_and_sentence_split(subj, sent, seed=3)
    b = subject_and_sentence_split(subj, sent, seed=3)
    assert np.array_equal(a.test, b.test)


def test_detects_shared_subject():
    subj = np.array([0, 0, 1, 1])
    sent = np.array([0, 1, 2, 3])
    leaky = Split(train=np.array([0, 2]), val=np.array([], dtype=int), test=np.array([1, 3]))
    with pytest.raises(LeakageError, match="subjects"):
        check_no_leakage(leaky, subj, sent, by=("subject",))


def test_detects_shared_sentence():
    subj = np.array([0, 1, 2, 3])
    sent = np.array([5, 5, 6, 7])
    leaky = Split(train=np.array([0, 2]), val=np.array([], dtype=int), test=np.array([1, 3]))
    with pytest.raises(LeakageError, match="sentences"):
        check_no_leakage(leaky, subj, sent, by=("sentence",))


def test_detects_shared_samples():
    subj, sent = np.array([0, 1, 2]), np.array([0, 1, 2])
    leaky = Split(train=np.array([0, 1]), val=np.array([], dtype=int), test=np.array([1, 2]))
    with pytest.raises(LeakageError, match="samples"):
        check_no_leakage(leaky, subj, sent)


def test_too_few_groups_is_an_error():
    with pytest.raises(ValueError):
        subject_split(np.array([0, 0, 1, 1]))


def test_calibration_is_disjoint_from_eval_sentences(ids):
    subj, sent = ids
    s = subject_and_sentence_split(subj, sent, seed=0)
    pool = np.concatenate([s.groups["train_sentences"], s.groups["val_sentences"]])
    person = int(s.groups["test_subjects"][0])
    cal = select_calibration(subj, sent, person, pool, n_sentences=3, seed=0)
    eval_idx = s.test[subj[s.test] == person]
    assert set(sent[cal]).isdisjoint(set(sent[eval_idx]))
    assert set(subj[cal]) == {person}
    assert len(set(sent[cal])) == 3


def test_calibration_keeps_all_trials_of_a_chosen_sentence(ids):
    subj, sent = ids
    pool = np.arange(20)
    cal = select_calibration(subj, sent, 0, pool, n_sentences=2, seed=0)
    assert len(cal) == 4  # 2 sentences x 2 trials


def test_zero_calibration_is_empty(ids):
    subj, sent = ids
    assert select_calibration(subj, sent, 0, np.arange(20), 0).size == 0


def test_calibration_asking_for_too_many_raises(ids):
    subj, sent = ids
    with pytest.raises(ValueError):
        select_calibration(subj, sent, 0, np.arange(3), n_sentences=10)


# ---- block-wise sentence splits (neighbouring sentences stay together) -----------------------
def test_block_split_keeps_every_block_in_one_partition():
    sent = np.repeat(np.arange(100), 3)
    s = sentence_split(sent, seed=2, block_size=10)
    check_no_leakage(s, np.zeros(len(sent), int), sent, by=("sentence",))
    part_of = {}
    for name in ("train_sentences", "val_sentences", "test_sentences"):
        for sid in s.groups[name]:
            part_of[int(sid)] = name
    for b in range(10):
        assert len({part_of[i] for i in range(b * 10, b * 10 + 10)}) == 1


def test_block_split_works_in_the_strict_two_factor_split_too():
    subj = np.repeat(np.arange(12), 40)
    sent = np.tile(np.arange(40), 12)
    s = subject_and_sentence_split(subj, sent, seed=1, block_size=5)
    check_no_leakage(s, subj, sent)
    test_sent = np.sort(s.groups["test_sentences"])
    assert set(test_sent // 5) and all(len(np.flatnonzero(test_sent // 5 == b)) == 5 for b in set(test_sent // 5))


def test_default_split_is_unchanged_by_the_new_option():
    sent = np.repeat(np.arange(60), 2)
    a, b = sentence_split(sent, seed=4), sentence_split(sent, seed=4, block_size=None)
    assert np.array_equal(a.test, b.test)


def test_block_size_must_be_positive():
    with pytest.raises(ValueError):
        sentence_split(np.arange(30), block_size=0)
