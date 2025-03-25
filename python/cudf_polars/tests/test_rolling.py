# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_ir_translation_raises


@pytest.fixture
def df():
    dates = pl.Series(
        [
            datetime.strptime("2020-01-01 13:45:48", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-01 16:42:13", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-01 16:45:09", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-02 18:12:48", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-03 19:45:32", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-08 23:16:43", "%Y-%m-%d %H:%M:%S"),
            datetime.strptime("2020-01-10 23:16:43", "%Y-%m-%d %H:%M:%S"),
        ],
        dtype=pl.Datetime(time_unit="us"),
    )
    return pl.LazyFrame(
        {
            "dt": dates,
            "values": [3, 7, 5, 9, 2, 1, 72],
            "floats": pl.Series(
                [float("nan"), 7, 5, 2, -10, 1, float("inf")], dtype=pl.Float64()
            ),
        }
    )


@pytest.mark.parametrize("closed", ["left", "right", "both", "none"])
@pytest.mark.parametrize("period", ["1w4d", "48h", "180s"])
def test_datetime_rolling(df, closed, period):
    q = df.rolling("dt", period=period, closed=closed).agg(
        sum_a=pl.sum("values"),
        min_a=pl.min("values"),
        max_a=pl.max("values"),
    )

    assert_ir_translation_raises(q, NotImplementedError)


def test_calendrical_period_unsupported(df):
    q = df.rolling("dt", period="1m", closed="right").agg(sum=pl.sum("values"))

    assert_ir_translation_raises(q, NotImplementedError)
