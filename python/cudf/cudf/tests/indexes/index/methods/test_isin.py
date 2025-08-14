# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_220
from cudf.testing import assert_eq
from cudf.testing._utils import expect_warning_if


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([]),
        pd.Index(["a", "b", "c", "d", "e"]),
        pd.Index([0, None, 9]),
        pd.date_range("2019-01-01", periods=3),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        [],
        ["this", "is"],
        [0, 19, 13],
        ["2019-01-01 04:00:00", "2019-01-01 06:00:00", "2018-03-02 10:00:00"],
    ],
)
def test_isin_index(index, values):
    pidx = index
    gidx = cudf.Index.from_pandas(pidx)

    is_dt_str = (
        next(iter(values), None) == "2019-01-01 04:00:00"
        and len(pidx)
        and pidx.dtype.kind == "M"
    )
    with expect_warning_if(is_dt_str):
        got = gidx.isin(values)
    with expect_warning_if(PANDAS_GE_220 and is_dt_str):
        expected = pidx.isin(values)

    assert_eq(got, expected)
