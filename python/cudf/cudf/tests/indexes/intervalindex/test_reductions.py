# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd

import cudf


def test_reduction_return_interval_pandas_compatible():
    ii = pd.IntervalIndex.from_tuples(
        [("2017-01-03", "2017-01-04")], dtype="interval[datetime64[ns], right]"
    )
    cudf_ii = cudf.IntervalIndex.from_pandas(ii)
    with cudf.option_context("mode.pandas_compatible", True):
        result = cudf_ii.min()
    expected = ii.min()
    assert result == expected
