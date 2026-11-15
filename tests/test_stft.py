"""Tests for the 'stft' feature set: output shape and column names for every
combination of window length and frequency-band width (stft_fs_window)."""
import numpy as np
import pytest
from scipy.signal import stft

import fsfa

T = 2000


def make_X(D, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((T, D)), axis=0)


def embed(X, window, stride, w):
    return fsfa.feature_embedding(X, window, stride, featureset="stft",
                                  stft_fs_window=w)


def test_default_band_width_runs():
    F = fsfa.feature_embedding(make_X(1), 500, 50, featureset="stft")
    assert F.shape == ((T - 500) // 50 + 1, 500 // 2 + 1)


# window -> n = window//2 + 1 frequency bins:
# 500 -> 251 (prime), 1000 -> 501, 38 -> 20, 64 -> 33
@pytest.mark.parametrize("window, w", [
    (500, 1), (500, 251),            # w divides n
    (38, 1), (38, 4), (38, 5),        # w divides n
    (38, 3), (38, 7), (38, 20),       # w does not divide n except 20
    (500, 20), (1000, 20), (64, 2),   # w does not divide n
    (1000, 1), (64, 33), (64, 11),    # w divides n
])
@pytest.mark.parametrize("D", [1, 3])
def test_shape_and_names(window, w, D):
    stride = max(1, window // 4)
    n_bins = window // 2 + 1
    n_bands = n_bins // w
    n_windows = (T - window) // stride + 1

    F = embed(make_X(D), window, stride, w)

    assert F.shape == (n_windows, D * n_bands)
    assert F.columns.is_unique
    for i in range(D):
        cols = F.columns[i * n_bands:(i + 1) * n_bands]
        assert all(c.startswith(f"x{i}_fx_") for c in cols)


@pytest.mark.parametrize("window, w", [(38, 4), (500, 1), (500, 251)])
def test_last_band_upper_edge_is_last_frequency(window, w):
    f = stft(np.zeros(window), 1.0, nperseg=window)[0]
    F = embed(make_X(1), window, window // 2, w)
    assert F.columns[-1] == f"x0_fx_{f[len(f) - w]}_{f[-1]}"


@pytest.mark.parametrize("window, w", [(500, 20), (1000, 20), (38, 3)])
def test_names_match_previous_formula(window, w):
    # cases where n is not divisible by w, for which the earlier naming worked
    D = 2
    f = stft(np.zeros(window), 1.0, nperseg=window)[0]
    assert len(f) % w != 0
    expected = [f"x{i}_fx_{a}_{b}" for i in range(D)
                for (a, b) in zip(f[::w], f[w::w])]

    F = embed(make_X(D), window, window // 4, w)

    assert list(F.columns) == expected
