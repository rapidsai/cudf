# Copyright (c) 2023-2024, NVIDIA CORPORATION.

import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_slice_datetimetz_index():
    data = ["2001-01-01", "2001-01-02", None, None, "2001-01-03"]
    pidx = pd.DatetimeIndex(data, dtype="datetime64[ns]").tz_localize(
        "US/Eastern"
    )
    idx = cudf.DatetimeIndex(data, dtype="datetime64[ns]").tz_localize(
        "US/Eastern"
    )
    expected = pidx[1:4]
    got = idx[1:4]
    assert_eq(expected, got)
