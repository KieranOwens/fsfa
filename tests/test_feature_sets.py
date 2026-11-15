"""Combining named feature sets, and choosing the number of 'stft' frequency bands."""
import numpy as np
import pytest

import fsfa

T, WINDOW, STRIDE = 3000, 500, 100
N_WINDOWS = (T - WINDOW) // STRIDE + 1
N_BINS = WINDOW // 2 + 1


@pytest.fixture(scope="module")
def X():
    rng = np.random.default_rng(0)
    return np.cumsum(rng.standard_normal((T, 2)), axis=0)


def embed(X, **kwargs):
    return fsfa.feature_embedding(X, window=WINDOW, stride=STRIDE, **kwargs)


##############################################################################
# combining named feature sets
##############################################################################

def test_combined_sets_join_the_columns_of_each(X):
    F = embed(X, featureset=['stft', 'quantiles'], stft_nb_bands=10, nb_quantiles=5)
    stft = embed(X, featureset='stft', stft_nb_bands=10)
    quantiles = embed(X, featureset='quantiles', nb_quantiles=5)

    assert list(F.columns) == list(stft.columns) + list(quantiles.columns)
    np.testing.assert_array_equal(F.to_numpy(),
                                  np.hstack([stft.to_numpy(), quantiles.to_numpy()]))


@pytest.mark.parametrize("names", [['catch22'], ('catch22',)])
def test_one_name_in_a_list_is_that_feature_set(X, names):
    F = embed(X, featureset=names)
    expected = embed(X, featureset='catch22')
    assert list(F.columns) == list(expected.columns)
    np.testing.assert_array_equal(F.to_numpy(), expected.to_numpy())


def test_combined_sets_pass_variable_names(X):
    F = embed(X, featureset=['mean', 'quantiles'], nb_quantiles=3, variablenames=['a', 'b'])
    assert list(F.columns) == ['a_mean', 'b_mean',
                               'a_q0', 'a_q1', 'a_q2', 'b_q0', 'b_q1', 'b_q2']


@pytest.mark.parametrize("names", [['mean', 'meanvar'], ['catch22', 'catch24'],
                                   ['mean', 'mean']])
def test_sets_with_features_in_common_raise(X, names):
    with pytest.raises(ValueError, match="in common"):
        embed(X, featureset=names)


def test_mixture_of_names_and_functions_raises(X):
    with pytest.raises(TypeError, match="mixture"):
        embed(X, featureset=['mean', np.var])


@pytest.mark.parametrize("featureset", ['catch23', ['mean', 'catch23']])
def test_unknown_name_raises(X, featureset):
    with pytest.raises(ValueError, match="Unknown feature set 'catch23'"):
        embed(X, featureset=featureset)


def test_fsfa_with_combined_sets(X):
    y = fsfa.fSFA(X, window=WINDOW, stride=STRIDE, featureset=['stft', 'quantiles'],
                  stft_nb_bands=10, n_components=2)
    assert y.shape == (N_WINDOWS, 2)
    assert np.isfinite(y).all()


##############################################################################
# number of stft frequency bands
##############################################################################

@pytest.mark.parametrize("nb_bands", [1, 2, 10, 25, N_BINS])
def test_nb_bands_shape_and_names(X, nb_bands):
    F = embed(X, featureset='stft', stft_nb_bands=nb_bands)

    assert F.shape == (N_WINDOWS, X.shape[1] * nb_bands)
    assert np.isfinite(F.to_numpy()).all()
    assert F.columns.is_unique
    # with fs = 1 the frequencies run from 0 to 0.5
    assert F.columns[0].startswith('x0_fx_0.0_')
    assert F.columns[nb_bands - 1].endswith('_0.5')
    assert F.columns[nb_bands].startswith('x1_fx_0.0_')


@pytest.mark.parametrize("nb_bands", [2, 10, 25])
def test_nb_bands_use_every_frequency(X, nb_bands):
    """The bands' proportional powers sum to the same total in every window."""
    F = embed(X[:, :1], featureset='stft', stft_nb_bands=nb_bands)
    # each feature is 20 * log(power / largest power in the spectrogram)
    total = np.exp(F.to_numpy() / 20).sum(axis=1)
    np.testing.assert_allclose(total, total[0])


def test_nb_bands_matches_fs_window_when_widths_are_equal(X):
    # a window of 38 samples has 20 frequencies: 5 bands of 4
    by_number = fsfa.feature_embedding(X, window=38, stride=19, featureset='stft',
                                       stft_nb_bands=5)
    by_width = fsfa.feature_embedding(X, window=38, stride=19, featureset='stft',
                                      stft_fs_window=4)
    assert list(by_number.columns) == list(by_width.columns)
    np.testing.assert_array_equal(by_number.to_numpy(), by_width.to_numpy())


@pytest.mark.parametrize("nb_bands", [0, -1, N_BINS + 1])
def test_nb_bands_out_of_range_raises(X, nb_bands):
    with pytest.raises(ValueError, match="stft_nb_bands must be between"):
        embed(X, featureset='stft', stft_nb_bands=nb_bands)


def test_nb_bands_and_fs_window_together_raise(X):
    with pytest.raises(ValueError, match="either"):
        embed(X, featureset='stft', stft_nb_bands=10, stft_fs_window=20)
