# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import datetime
from operator import methodcaller

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Date(),
        pl.Datetime("ms"),
        pl.Datetime("us"),
        pl.Datetime("ns"),
        pl.Duration("ms"),
        pl.Duration("us"),
        pl.Duration("ns"),
    ],
    ids=repr,
)
def test_datetime_dataframe_scan(dtype):
    ldf = pl.DataFrame(
        {
            "a": pl.Series([1, 2, 3, 4, 5, 6, 7], dtype=dtype),
            "b": pl.Series([3, 4, 5, 6, 7, 8, 9], dtype=pl.UInt16),
        }
    ).lazy()

    query = ldf.select(pl.col("b"), pl.col("a"))
    assert_gpu_result_equal(query)


@pytest.mark.parametrize(
    "field",
    [
        methodcaller("year"),
        pytest.param(
            methodcaller("day"),
            marks=pytest.mark.xfail(reason="day extraction not implemented"),
        ),
    ],
)
def test_datetime_extract(field):
    ldf = pl.LazyFrame(
        {"dates": [datetime.date(2024, 1, 1), datetime.date(2024, 10, 11)]}
    )
    q = ldf.select(field(pl.col("dates").dt))

    with pytest.raises(AssertionError):
        # polars produces int32, libcudf produces int16 for the year extraction
        # libcudf can lose data here.
        # https://github.com/rapidsai/cudf/issues/16196
        assert_gpu_result_equal(q)

    assert_gpu_result_equal(q, check_dtypes=False)
