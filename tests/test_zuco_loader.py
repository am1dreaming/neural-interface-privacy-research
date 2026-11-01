import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from neurotext.data import EEGData, load_zuco2_nr, read_subject_file, tfidf_svd_embeddings
from neurotext.data import zuco as zuco_module

SENTENCES = [
    "Henry Ford founded the Ford Foundation in 1936.",
    "After this initial success Ford left Edison Illuminating.",
    "Jeb Bush was born in Midland Texas.",
    "The quick brown fox jumps over the lazy dog.",
    "Marie Curie won two Nobel prizes in physics and chemistry.",
    "Mount Everest is the highest mountain above sea level.",
]


def make_fake_mat(path, sentences, lengths, n_words, nan_at=(), seed=0):
    """Same layout as a ZuCo 2.0 MATLAB v7.3 file: sentenceData fields are (n, 1) arrays of
    object references into #refs#; rawData is (samples, 105); a missing recording is a scalar."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        sd = f.create_group("sentenceData")
        ds = {k: sd.create_dataset(k, (len(sentences), 1), dtype=h5py.ref_dtype)
              for k in ("content", "rawData", "wordbounds")}
        for i, s in enumerate(sentences):
            c = refs.create_dataset(f"c{i}", data=np.array([[ord(ch)] for ch in s], dtype=np.uint16))
            ds["content"][i, 0] = c.ref
            if lengths[i] is None:
                raw = np.zeros((1, 1))
            else:
                raw = rng.standard_normal((lengths[i], 105)) * 5 + 3
                if i in nan_at:
                    raw[10, 3] = np.nan
            ds["rawData"][i, 0] = refs.create_dataset(f"r{i}", data=raw).ref
            ds["wordbounds"][i, 0] = refs.create_dataset(f"w{i}", data=np.zeros((4, n_words[i]))).ref
    return path


@pytest.fixture
def one_file(tmp_path):
    #            ok    missing  too short  ok(long)  ok    has NaN
    lengths = [1000, None, 40, 4000, 800, 1200]
    return make_fake_mat(tmp_path / "resultsABC_NR.mat", SENTENCES, lengths, [10] * 6, nan_at={5})


def test_filters_are_applied_and_counted(one_file):
    r = read_subject_file(one_file)
    assert r.code == "ABC"
    assert r.sentence_idx.tolist() == [0, 3, 4]
    assert r.dropped == {"no_eeg": 1, "too_short": 1, "non_finite": 1}


def test_texts_are_decoded_in_file_order(one_file):
    assert read_subject_file(one_file).sentences == SENTENCES


def test_windows_have_fixed_shape_and_dtype(one_file):
    r = read_subject_file(one_file, decimate_factor=5, n_times=640)
    assert r.x.shape == (3, 105, 640) and r.x.dtype == np.float32


def test_short_recordings_are_zero_padded_long_ones_cropped(one_file):
    x = read_subject_file(one_file, decimate_factor=5, n_times=640).x
    # sentence 0: 1000 samples -> 200 after /5, the rest must be padding
    assert np.all(x[0, :, 200:] == 0) and np.any(x[0, :, :200] != 0)
    # sentence 3: 4000 samples -> 800 -> cropped to the first 640, no padding
    assert np.all(np.abs(x[1, :, -1]) > 0)


def test_real_samples_are_zscored_per_channel(one_file):
    x = read_subject_file(one_file, decimate_factor=5, n_times=640).x[0, :, :200]
    assert np.allclose(x.mean(axis=1), 0, atol=1e-4) and np.allclose(x.std(axis=1), 1, atol=1e-3)


def test_downsampling_removes_content_above_the_new_nyquist(tmp_path):
    t = np.arange(2000) / 500.0
    low, high = np.sin(2 * np.pi * 5 * t), np.sin(2 * np.pi * 150 * t)  # 5 Hz kept, 150 Hz > 50 Hz Nyquist
    p = tmp_path / "resultsHI_NR.mat"
    make_fake_mat(p, SENTENCES[:3], [2000] * 3, [10] * 3)
    with h5py.File(p, "r+") as f:  # replace sentence 0 with "5 Hz + 150 Hz" on every channel
        del f["#refs#/r0"]
        r = f["#refs#"].create_dataset("r0", data=np.tile((low + high)[:, None], (1, 105)))
        f["sentenceData/rawData"][0, 0] = r.ref
    x = read_subject_file(p, decimate_factor=5, n_times=400).x[0, 0]
    ref = zuco_module._prepare_window(np.tile(low[None, :], (105, 1)), 5, 400)[0]
    assert np.corrcoef(x, ref)[0, 1] > 0.95  # the 150 Hz part was filtered out, not aliased in


def test_cache_is_used_on_second_read(one_file, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    a = read_subject_file(one_file, cache_dir=cache)
    monkeypatch.setattr(zuco_module, "_h5py", lambda: (_ for _ in ()).throw(RuntimeError("file was read again")))
    b = read_subject_file(one_file, cache_dir=cache)
    assert np.array_equal(a.x, b.x) and a.sentences == b.sentences and a.dropped == b.dropped
    assert list(cache.glob("*.npz")) and not list(cache.glob("*.tmp"))


def test_cache_key_depends_on_preprocessing_options(one_file, tmp_path):
    cache = tmp_path / "cache"
    read_subject_file(one_file, cache_dir=cache, n_times=640)
    read_subject_file(one_file, cache_dir=cache, n_times=320)
    assert len(list(cache.glob("*.npz"))) == 2


def _two_people(tmp_path):
    make_fake_mat(tmp_path / "resultsBBB_NR.mat", SENTENCES, [1000] * 6, [10] * 6, seed=2)
    make_fake_mat(tmp_path / "resultsAAA_NR.mat", SENTENCES, [1000, None, 1500, 1200, 900, 2000], [10] * 6, seed=1)


def test_loading_several_people_gives_the_common_interface(tmp_path):
    _two_people(tmp_path)
    d = load_zuco2_nr(tmp_path, text_dim=4)
    assert isinstance(d, EEGData)
    assert d.subject_codes == ["AAA", "BBB"]                      # sorted, ids follow this order
    assert d.x.shape == (11, 105, 640) and d.sfreq == 100.0
    assert (d.subject_ids == 0).sum() == 5 and (d.subject_ids == 1).sum() == 6
    assert sorted(d.sentence_ids[d.subject_ids == 0].tolist()) == [0, 2, 3, 4, 5]
    assert d.text_emb.shape[0] == 6 and np.allclose(np.linalg.norm(d.text_emb, axis=1), 1, atol=1e-5)
    assert d.meta["dropped"]["AAA"]["no_eeg"] == 1 and d.meta["kept"] == {"AAA": 5, "BBB": 6}
    assert d.meta["window_seconds"] == pytest.approx(6.4)


def test_subject_selection(tmp_path):
    _two_people(tmp_path)
    assert load_zuco2_nr(tmp_path, subjects=["BBB"], text_dim=4).subject_codes == ["BBB"]
    assert load_zuco2_nr(tmp_path, max_subjects=1, text_dim=4).subject_codes == ["AAA"]
    with pytest.raises(ValueError, match="not found"):
        load_zuco2_nr(tmp_path, subjects=["ZZZ"])


def test_people_with_different_sentence_lists_are_rejected(tmp_path):
    make_fake_mat(tmp_path / "resultsAAA_NR.mat", SENTENCES, [1000] * 6, [10] * 6)
    other = SENTENCES[:5] + ["A completely different sentence about trains."]
    make_fake_mat(tmp_path / "resultsBBB_NR.mat", other, [1000] * 6, [10] * 6)
    with pytest.raises(ValueError, match="different sentences"):
        load_zuco2_nr(tmp_path, text_dim=4)


def test_missing_data_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_zuco2_nr(tmp_path)


def test_partial_download_files_are_ignored(tmp_path):
    _two_people(tmp_path)
    (tmp_path / "resultsCCC_NR.mat.part").write_bytes(b"incomplete")
    assert load_zuco2_nr(tmp_path, text_dim=4).subject_codes == ["AAA", "BBB"]


def test_text_embeddings_are_deterministic_unit_norm_and_lexically_sensible():
    a = tfidf_svd_embeddings(SENTENCES + ["Ford founded a foundation in Detroit."], dim=4)
    b = tfidf_svd_embeddings(SENTENCES + ["Ford founded a foundation in Detroit."], dim=4)
    assert np.array_equal(a, b) and np.allclose(np.linalg.norm(a, axis=1), 1, atol=1e-5)
    sim = a @ a.T
    assert sim[0, 6] > sim[0, 3]  # the Ford sentence is closer to the Ford sentence than to the fox


def test_eegdata_rejects_inconsistent_arrays():
    with pytest.raises(ValueError):
        EEGData(np.zeros((3, 2, 4), np.float32), np.zeros(3, int), np.zeros(2, int), np.zeros((5, 2), np.float32))
    with pytest.raises(ValueError, match="past the end"):
        EEGData(np.zeros((2, 2, 4), np.float32), np.zeros(2, int), np.array([0, 9]), np.zeros((5, 2), np.float32))


def test_real_data_path_runs_through_the_whole_experiment(tmp_path):
    pytest.importorskip("torch")
    from neurotext.experiment import run_once

    sents = [f"Sentence number {k} talks about topic{k} and another thing{k} entirely." for k in range(24)]
    for s in range(6):
        make_fake_mat(tmp_path / f"resultsS{s:02d}_NR.mat", sents, [1000] * 24, [10] * 24, seed=s)
    cfg = {
        "data": {"source": "zuco2_nr", "root": str(tmp_path), "cache_dir": str(tmp_path / "cache"),
                 "decimate_factor": 5, "n_times": 64, "min_seconds_per_word": 0.1, "text_dim": 8},
        "split": {"val_frac": 0.15, "test_frac": 0.25},
        "model": {"d_model": 16, "out_dim": 8, "n_layers": 1, "n_heads": 2, "patch": 16},
        "train": {"epochs": 2, "lr": 0.001, "batch_size": 32, "temperature": 0.1},
        "personalization": {"rank": 2, "epochs": 2, "lr": 0.001, "calibration_sentences": [0, 2]},
    }
    r = run_once(cfg, seed=0, include_efficiency=False)
    assert r["synthetic"] is False and r["n_candidates"] == 6 and set(r["calibration_curve_top1"]) == {0, 2}
