import numpy as np
import pytest

from neurotext.features import (
    GROUPS, amplitude_normalise, feature_table, region_of, trim_edges, welch_psd,
)


def sines(freqs, fs=100, n=600, c=3, noise=0.05, seed=0, amp=1.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    x = np.stack([[amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) for _ in range(c)] for f in freqs])
    return (x + noise * rng.standard_normal(x.shape)).astype(np.float32)


def test_welch_peak_is_at_the_planted_frequency():
    x = sines([10.0], n=800)
    f, p = welch_psd(x, 100)
    assert abs(f[np.argmax(p[0, 0])] - 10.0) <= f[1] - f[0]


def test_alpha_peak_features_recover_each_persons_peak_frequency():
    peaks = [8.5, 9.5, 10.5, 11.5]
    x = sines(peaks)
    ft = feature_table(x, 100)
    freq_cols = ft.groups["alpha_peak"][:3]  # first block of the group = peak frequency of each channel
    got = ft.x[:, freq_cols].mean(axis=1)
    assert np.abs(got - peaks).max() < 0.5


def test_groups_cover_columns_once_and_columns_selector_works():
    ft = feature_table(sines([10.0] * 5), 100)
    assert set(ft.groups) <= set(GROUPS)
    allcols = np.sort(np.concatenate(list(ft.groups.values())))
    assert np.array_equal(allcols, np.arange(ft.x.shape[1]))
    only = ft.columns(only=("rms",))
    assert len(only) == 3 and set(ft.channel[only]) == {0, 1, 2}
    rest = ft.columns(drop=("rms",))
    assert len(rest) == ft.x.shape[1] - 3 and not set(rest) & set(only)
    with pytest.raises(ValueError, match="unknown feature groups"):
        ft.columns(only=("nonsense",))


def test_bands_the_sampling_rate_cannot_resolve_are_left_out():
    ft = feature_table(sines([10.0], fs=40, n=400), 40)
    assert "gamma" not in ft.groups and "alpha" in ft.groups


def test_amplitude_normalisation_removes_gain_but_keeps_spectrum_shape():
    x = sines([6.0, 12.0])
    scaled = x * 7.0
    assert np.allclose(amplitude_normalise(x), amplitude_normalise(scaled), atol=1e-4)
    ft = feature_table(amplitude_normalise(scaled), 100)
    assert np.allclose(ft.x[:, ft.groups["rms"]], 0, atol=1e-4)           # amplitude is gone
    assert ft.x[0, ft.groups["alpha"][0]] != ft.x[1, ft.groups["alpha"][0]]  # spectral shape is not


def test_trim_edges():
    x = np.zeros((2, 3, 100))
    assert trim_edges(x, 0.1).shape == (2, 3, 80) and trim_edges(x, 0).shape == x.shape
    with pytest.raises(ValueError):
        trim_edges(x, 0.5)


@pytest.mark.parametrize("name,region", [("Fc5.", "central"), ("Cz..", "central"), ("Oz..", "occipital"),
                                         ("Fp1.", "frontal"), ("T7..", "temporal"), ("P3..", "parietal"),
                                         ("Iz..", "occipital"), ("Cpz.", "parietal"), ("Af7.", "frontal"),
                                         ("Po7.", "occipital"), ("Ft8.", "temporal"), ("???", "other")])
def test_region_of(name, region):
    assert region_of(name) == region
