# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_nunique_all_null(dropna):
    data = [None, None]
    pd_ser = pd.Series(data)
    cudf_ser = cudf.Series(data)
    result = pd_ser.nunique(dropna=dropna)
    expected = cudf_ser.nunique(dropna=dropna)
    assert result == expected


def test_series_nunique():
    cd_s = cudf.Series([1, 3, 5, 7, 7])
    pd_s = cd_s.to_pandas()

    actual = cd_s.nunique()
    expected = pd_s.nunique()

    assert_eq(expected, actual)
