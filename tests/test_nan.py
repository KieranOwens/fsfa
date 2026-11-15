"""catch22 features that cannot be estimated come back as NaN, on any pycatch22 version."""
import warnings

import numpy as np
import pycatch22

import fsfa

# On pycatch22 < 1.0 these three return 0 (not NaN) on a constant window.
ZERO_ON_CONSTANT = {'acf_timescale', 'acf_first_min', 'periodicity'}

NAMES = fsfa.FEATURE_DICT['catch22']['names']
FUNCS = fsfa.FEATURE_DICT['catch22']['features']


def _values(x):
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        return dict(zip(NAMES, (f(list(x)) for f in FUNCS)))


def test_constant_window_gives_nan():
    values = _values(np.ones(200))
    assert len(values) == 22
    for name, v in values.items():
        if name in ZERO_ON_CONSTANT:
            assert np.isnan(v) or v == 0, (name, v)
        else:
            assert np.isnan(v), (name, v)


def test_random_walk_all_finite():
    x = np.cumsum(np.random.default_rng(0).standard_normal(500))
    values = _values(x)
    assert len(values) == 22
    assert all(np.isfinite(v) for v in values.values()), values


def test_random_walk_matches_default_pycatch22():
    """On an ordinary window, the values equal pycatch22's own (default) output."""
    x = list(np.cumsum(np.random.default_rng(1).standard_normal(500)))
    values = list(_values(x).values())
    if hasattr(pycatch22, 'feature_table'):
        expected = [getattr(pycatch22, f.__name__)(x) for f in FUNCS]
    else:
        out = pycatch22.catch22_all(x)
        by_name = dict(zip(out['names'], out['values']))
        expected = [by_name[f.__name__] for f in FUNCS]
    np.testing.assert_allclose(values, expected)
