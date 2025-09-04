# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd

import cudf


def test_nunique():
    gidx = cudf.RangeIndex(5)
    pidx = pd.RangeIndex(5)

    actual = gidx.nunique()
    expected = pidx.nunique()

    assert actual == expected
