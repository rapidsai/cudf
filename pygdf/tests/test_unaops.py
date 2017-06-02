from __future__ import division

import numpy as np

from pygdf.dataframe import Series


def test_series_ceil():
    arr = np.random.random(100)
    sr = Series.from_any(arr)
    np.testing.assert_equal(sr.ceil().to_array(), np.ceil(arr))


def test_series_floor():
    arr = np.random.random(100)
    sr = Series.from_any(arr)
    np.testing.assert_equal(sr.floor().to_array(), np.floor(arr))
