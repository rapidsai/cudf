import pytest

import numpy as np

from numba import cuda

from pygdf.dataframe import DataFrame, Series


def test_series_equal():
    arr = np.random.random(100)
    sr = Series.from_any(arr)
    sr_eq_sr = (sr == sr).to_array()
    assert np.all(sr_eq_sr)
    sr_ne_sr = (sr != sr).to_array()
    assert not np.any(sr_ne_sr)


def test_series_less():
    s1 = Series.from_any(np.arange(10, dtype=np.float32))
    s2 = Series.from_any(np.arange(10, dtype=np.float32) * 2)

    s1_lt_s2 = (s1 < s2).to_array()
    assert np.all(s1_lt_s2[1:])
    assert not s1_lt_s2[0]

    s1_le_s2 = (s1 <= s2).to_array()
    assert np.all(s1_le_s2[1:])
    assert s1_le_s2[0]


def test_series_greater():
    s1 = Series.from_any(np.arange(10, dtype=np.float32))
    s2 = Series.from_any(np.arange(10, dtype=np.float32) * 2)

    s1_gt_s2 = (s1 > s2).to_array()
    assert not np.any(s1_gt_s2[1:])
    assert not s1_gt_s2[0]

    s1_ge_s2 = (s1 >= s2).to_array()
    assert not np.any(s1_ge_s2[1:])
    assert s1_ge_s2[0]
