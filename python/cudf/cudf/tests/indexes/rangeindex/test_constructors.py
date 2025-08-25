# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_from_pandas_rangeindex():
    idx1 = pd.RangeIndex(start=0, stop=4, step=1, name="myindex")
    idx2 = cudf.from_pandas(idx1)

    # Check index
    assert_eq(idx1.values, idx2.values)
    assert idx1.name == idx2.name


def test_from_pandas_rangeindex_step():
    expected = pd.RangeIndex(start=0, stop=8, step=2, name="myindex")
    actual = cudf.from_pandas(expected)

    assert_eq(expected, actual)
