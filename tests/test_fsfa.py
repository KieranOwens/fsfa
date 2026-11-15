"""Smoke tests for fsfa: every built-in feature set runs, and the catch22 functions
that fsfa looks up are the ones pycatch22 computes, in the order fsfa names them."""
import warnings

import numpy as np
import pycatch22
import pytest

import fsfa

T, WINDOW, STRIDE = 3000, 500, 100
N_WINDOWS = (T - WINDOW) // STRIDE + 1


@pytest.fixture(scope="module")
def X():
    rng = np.random.default_rng(0)
    return np.cumsum(rng.standard_normal((T, 2)), axis=0)


def test_names_match_features():
    for name, entry in fsfa.FEATURE_DICT.items():
        assert len(entry["names"]) == len(entry["features"]), name
    assert len(fsfa.FEATURE_DICT["catch22"]["features"]) == 22
    assert len(fsfa.FEATURE_DICT["catch24"]["features"]) == 24


@pytest.mark.parametrize("featureset, n_per_variable", [
    ("mean", 1),
    ("meanvar", 2),
    ("catch22", 22),
    ("catch24", 24),
    ("quantiles", 20),
])
def test_feature_embedding_shape(X, featureset, n_per_variable):
    F = fsfa.feature_embedding(X, window=WINDOW, stride=STRIDE, featureset=featureset)
    assert F.shape == (N_WINDOWS, n_per_variable * X.shape[1])
    assert np.isfinite(F.to_numpy()).all()
    assert len(set(F.columns)) == F.shape[1]


def test_feature_embedding_stft(X):
    F = fsfa.feature_embedding(X, window=WINDOW, stride=STRIDE, featureset="stft",
                               stft_fs_window=20)
    assert F.shape[0] == N_WINDOWS
    assert np.isfinite(F.to_numpy()).all()


def test_feature_embedding_custom_functions(X):
    F = fsfa.feature_embedding(X, window=WINDOW, stride=STRIDE,
                               featureset=[np.mean, np.median],
                               featurenames=["mean", "median"],
                               variablenames=["a", "b"])
    assert list(F.columns) == ["a_mean", "a_median", "b_mean", "b_median"]
    np.testing.assert_allclose(F["a_mean"][0], X[:WINDOW, 0].mean())


def test_catch22_functions_match_pycatch22(X):
    """Each function fsfa uses gives the value catch22_all gives for that feature."""
    x = list(X[:WINDOW, 0])
    result = pycatch22.catch22_all(x)
    expected = dict(zip(result["names"], result["values"]))
    if hasattr(pycatch22, "feature_table"):
        # pycatch22 >= 1.0 reports the features under their v1.0 names
        current = {f["legacy_name"]: f["name"] for f in pycatch22.feature_table()}
    else:
        current = {name: name for name in fsfa.features_c22}

    functions = fsfa.FEATURE_DICT["catch22"]["features"]
    for legacy_name, function in zip(fsfa.features_c22, functions):
        np.testing.assert_allclose(function(x), expected[current[legacy_name]],
                                   err_msg=legacy_name)


def test_catch22_raises_no_deprecation_warning(X):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        fsfa.feature_embedding(X, window=WINDOW, stride=STRIDE, featureset="catch22")


@pytest.mark.parametrize("n_components", [1, 2])
def test_fsfa(X, n_components):
    y = fsfa.fSFA(X, window=WINDOW, stride=STRIDE, featureset="catch22",
                  n_components=n_components)
    assert y.shape == (N_WINDOWS, n_components)
    assert np.isfinite(y).all()
