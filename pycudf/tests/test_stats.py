import pytest

import numpy as np

from numba import cuda

from pycudf.dataframe import DataFrame, Series


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


if __name__ == '__main__':
    test_series_var()

