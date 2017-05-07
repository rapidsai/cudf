import pytest

import numpy as np

from numba import cuda

from pygdf.dataframe import DataFrame, Series


def test_series_min():
    arr = np.random.random(100)
    sr = Series.from_any(arr)
    np.testing.assert_almost_equal(arr.min(), sr.min())


def test_series_max():
    arr = np.random.random(100)
    sr = Series.from_any(arr)
    np.testing.assert_almost_equal(arr.max(), sr.max())


def test_series_mean():
    arr = np.random.random(100)
    sr = Series.from_any(arr)
    np.testing.assert_almost_equal(arr.mean(), sr.mean())


def test_series_var():
    arr = np.random.random(100)
    sr = Series.from_any(arr)
    np.testing.assert_almost_equal(arr.var(), sr.var())


def test_series_std():
    arr = np.random.random(100)
    sr = Series.from_any(arr)
    np.testing.assert_almost_equal(arr.std(), sr.std())


def test_series_unique():
    for size in [10 ** x for x in range(5)]:
        arr = np.random.randint(low=0, high=10, size=size)
        sr = Series.from_any(arr)
        assert set(arr) == set(sr.unique_k(k=10))
    # test out of space
    arr = np.arange(10)
    sr = Series.from_any(arr)
    with pytest.raises(ValueError) as raises:
        sr.unique_k(k=7)
    raises.match('too many unique value')


def test_series_scale():
    arr = np.random.randint(low=-10, high=10, size=100)
    sr = Series.from_any(arr)

    vmin = arr.min()
    vmax = arr.max()
    scaled = (arr - vmin) / (vmax - vmin)
    assert scaled.min() == 0
    assert scaled.max() == 1
    np.testing.assert_equal(sr.scale().to_array(), scaled)


if __name__ == '__main__':
    test_series_scale()

