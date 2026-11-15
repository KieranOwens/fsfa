"""fsfa's catch22/catch24 column labels name the features whose values they hold."""
import numpy as np
import pycatch22
import pytest

import fsfa

T = 2000

# Short names that catch22 itself renamed: fsfa keeps its original label.
KNOWN_RENAMES = {
    'transition_matrix': 'transition_variance',
    'centroid_freq': 'median_freq',
}


@pytest.fixture(scope="module")
def noise():
    return np.random.default_rng(1).standard_normal(T)


@pytest.fixture(scope="module")
def pycatch22_by_legacy_name(noise):
    """pycatch22's (short name, value) for each catch22 feature, keyed by pre-v1.0 name."""
    out = pycatch22.catch22_all(noise, short_names=True)
    names = out['names']
    if hasattr(pycatch22, 'feature_table'):
        legacy = {f['name']: f['legacy_name'] for f in pycatch22.feature_table()}
        names = [legacy[name] for name in names]
    return {name: (short, value)
            for name, short, value in zip(names, out['short_names'], out['values'])}


def test_features_c22_order_and_names():
    assert len(fsfa.features_c22) == 22
    assert fsfa.features_c22 == [name for name, _ in fsfa.CATCH22_FEATURES]
    assert fsfa.FEATURE_DICT['catch22']['names'] == [s for _, s in fsfa.CATCH22_FEATURES]
    assert fsfa.FEATURE_DICT['catch24']['names'][:22] == fsfa.FEATURE_DICT['catch22']['names']


def test_short_names_match_pycatch22(pycatch22_by_legacy_name):
    assert set(pycatch22_by_legacy_name) == set(fsfa.features_c22)
    for name, short in fsfa.CATCH22_FEATURES:
        theirs = pycatch22_by_legacy_name[name][0]
        assert short == theirs or KNOWN_RENAMES.get(short) == theirs, (name, short, theirs)


def _single_window(x, featureset):
    """fsfa features of x computed over one window covering the whole series."""
    F = fsfa.feature_embedding(x[:, None], window=len(x), stride=len(x), featureset=featureset)
    assert F.shape[0] == 1
    return F.iloc[0]


def test_labelled_values_match_pycatch22(noise, pycatch22_by_legacy_name):
    row = _single_window(noise, 'catch22')
    for name, short in fsfa.CATCH22_FEATURES:
        expected = pycatch22_by_legacy_name[name][1]
        assert row[f'x0_{short}'] == pytest.approx(expected, rel=1e-9, abs=1e-12), name


def test_spectral_labels_on_white_noise(noise):
    # White noise has about a fifth of its power in the lowest fifth of frequencies,
    # and its spectral centroid (or median frequency) is near pi/2.
    row = _single_window(noise, 'catch22')
    assert 0.1 < row['x0_low_freq_power'] < 0.35
    assert 1.2 < row['x0_centroid_freq'] < 2.0


def test_catch24_mean_and_variance(noise):
    F = fsfa.feature_embedding(noise[:, None], window=500, stride=250, featureset='catch24')
    assert list(F.columns[-2:]) == ['x0_mean', 'x0_variance']
    starts = range(0, T - 500 + 1, 250)
    np.testing.assert_allclose(F['x0_mean'], [np.mean(noise[s:s + 500]) for s in starts])
    np.testing.assert_allclose(F['x0_variance'], [np.var(noise[s:s + 500]) for s in starts])
