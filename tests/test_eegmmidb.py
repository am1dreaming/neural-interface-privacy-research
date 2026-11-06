import numpy as np
import pytest

from neurotext.data import eegmmidb as E


def write_edf(path, data, fs, annotations=True):
    """Minimal EDF+ writer for tests: 1 s records, int16 samples, +-3200 uV mapped to the int16 range."""
    n_ch, n = data.shape
    per = int(fs)
    n_rec = n // per
    chans = n_ch + (1 if annotations else 0)
    labels = [f"Ch{i}." for i in range(n_ch)] + (["EDF Annotations"] if annotations else [])
    pad = lambda s, w: str(s).ljust(w)[:w].encode("ascii")
    head = (pad("0", 8) + pad("X", 80) + pad("X", 80) + pad("01.01.00", 8) + pad("00.00.00", 8)
            + pad(256 * (chans + 1), 8) + pad("EDF+C", 44) + pad(n_rec, 8) + pad(1, 8) + pad(chans, 4))
    assert len(head) == 256
    nsamp = [per] * n_ch + ([60] if annotations else [])
    sig = (b"".join(pad(l, 16) for l in labels) + pad("", 80) * chans + pad("uV", 8) * chans
           + (pad(-3200, 8) * n_ch + (pad(-1, 8) if annotations else b"")) + (pad(3200, 8) * n_ch + (pad(1, 8) if annotations else b""))
           + pad(-32768, 8) * chans + pad(32767, 8) * chans + pad("", 80) * chans
           + b"".join(pad(k, 8) for k in nsamp) + pad("", 32) * chans)
    gain = 6400.0 / 65535.0
    dig = np.clip(np.round((data + 3200.0) / gain - 32768.0), -32768, 32767).astype("<i2")
    body = []
    for r in range(n_rec):
        for c in range(n_ch):
            body.append(dig[c, r * per : (r + 1) * per].tobytes())
        if annotations:
            body.append(np.zeros(60, "<i2").tobytes())
    path.write_bytes(head + sig + b"".join(body))


def make_root(tmp_path, codes, n_seconds=20, fs=160, n_ch=64, bad=None, seed=0):
    rng = np.random.default_rng(seed)
    for code in codes:
        (tmp_path / code).mkdir()
        for run in range(1, 15):
            f = fs
            if bad and bad.get((code, run)) == "rate":
                f = 128
            if bad and bad.get((code, run)) == "missing":
                continue
            write_edf(tmp_path / code / f"{code}R{run:02d}.edf", 20 * rng.standard_normal((n_ch, n_seconds * f)), f)
    return tmp_path


def test_read_edf_round_trip_and_annotation_channel_is_skipped(tmp_path):
    rng = np.random.default_rng(0)
    data = 50 * rng.standard_normal((5, 160 * 3))
    write_edf(tmp_path / "a.edf", data, 160)
    labels, fs, got = E.read_edf(tmp_path / "a.edf")
    assert fs == 160.0 and labels == [f"Ch{i}." for i in range(5)]
    assert got.shape == data.shape
    assert np.abs(got - data).max() < 0.06  # half a quantisation step is 0.049 uV


def test_read_edf_rejects_other_files(tmp_path):
    (tmp_path / "x.edf").write_bytes(b"not an edf file" * 40)
    with pytest.raises(ValueError, match="not an EDF"):
        E.read_edf(tmp_path / "x.edf")


def test_decimate_removes_what_the_new_rate_cannot_hold():
    fs, n = 160, 1600
    t = np.arange(n) / fs
    x = np.stack([np.sin(2 * np.pi * 5 * t), np.sin(2 * np.pi * 70 * t)]).astype(np.float32)
    y = E.decimate(x, 2)
    assert y.shape == (2, n // 2)
    mid = slice(100, -100)
    assert y[0, mid].std() > 0.65          # 5 Hz survives (rms of a unit sine is 0.707)
    assert y[1, mid].std() < 0.05          # 70 Hz would alias to 10 Hz; it is filtered out first


def test_window_starts_are_non_overlapping_even_and_bounded():
    s = E.window_starts(n_samples=3200, win=320, max_windows=6)
    assert len(s) == 6 and (np.diff(s) >= 320).all() and s[-1] + 320 <= 3200
    assert len(E.window_starts(500, 320, 6)) == 1
    assert len(E.window_starts(100, 320, 6)) == 0


def test_holdout_is_the_documented_rule_and_has_24_people():
    perm = np.random.default_rng(E.HOLDOUT_SEED).permutation(len(E.ALL_CODES))
    assert tuple(sorted(E.ALL_CODES[i] for i in perm[:24])) == E.HOLDOUT
    assert len(set(E.HOLDOUT)) == 24 and set(E.HOLDOUT) <= set(E.ALL_CODES)


def test_text_embeddings_are_unit_norm_and_runs_of_one_type_are_closer():
    t = E.text_embeddings()
    assert t.shape[0] == 14 and np.allclose(np.linalg.norm(t, axis=1), 1, atol=1e-6)
    same = t[2] @ t[6]      # runs 3 and 7: both fist execution
    other = t[2] @ t[3]     # runs 3 and 4: execution vs imagery
    assert same > other


def test_loader_excludes_the_held_out_people_unless_asked(tmp_path):
    root = make_root(tmp_path, ["S001", "S002", "S003"])  # S002 is held out
    d = E.load_eegmmidb(root)
    assert d.subject_codes == ["S001", "S003"]
    assert d.x.shape[1:] == (64, 320) and d.sfreq == 80.0
    assert set(np.unique(d.sentence_ids)) == set(range(14))
    assert E.load_eegmmidb(root, holdout="only").subject_codes == ["S002"]
    both = E.load_eegmmidb(root, holdout="include")
    assert both.subject_codes == ["S001", "S002", "S003"]
    assert list(E.holdout_ids(both)) == [1]
    with pytest.raises(ValueError, match="holdout must be"):
        E.load_eegmmidb(root, holdout="maybe")


def test_loader_drops_bad_runs_and_people_with_too_few_runs(tmp_path):
    bad = {("S001", 3): "rate", ("S001", 4): "missing"}
    bad.update({("S003", r): "missing" for r in range(1, 6)})  # S003 keeps 9 of 14 runs
    root = make_root(tmp_path, ["S001", "S003"], bad=bad)
    d = E.load_eegmmidb(root, min_runs=12)
    assert d.subject_codes == ["S001"]
    assert d.meta["dropped"]["S001"] == {"other_sampling_rate": 1, "missing": 1}
    assert "S003" in d.meta["dropped_subjects"]
    assert 2 not in d.sentence_ids and 3 not in d.sentence_ids


def test_loader_cache_gives_identical_data(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    root = make_root(raw, ["S001"])
    a = E.load_eegmmidb(root, cache_dir=tmp_path / "cache")
    b = E.load_eegmmidb(root, cache_dir=tmp_path / "cache")
    assert np.array_equal(a.x, b.x) and np.array_equal(a.sentence_ids, b.sentence_ids)


def test_missing_root_has_a_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        E.load_eegmmidb(tmp_path / "nope")
