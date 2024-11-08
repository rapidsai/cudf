# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import datetime
from operator import methodcaller

import pytest

import polars as pl

from cudf_polars.dsl.expr import TemporalFunction
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


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


datetime_extract_fields = [
    "year",
    "month",
    "day",
    "weekday",
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
]


@pytest.fixture(
    ids=datetime_extract_fields,
    params=[methodcaller(f) for f in datetime_extract_fields],
)
def field(request):
    return request.param


def test_datetime_extract(field):
    ldf = pl.LazyFrame(
        {
            "datetimes": pl.datetime_range(
                datetime.datetime(2020, 1, 1),
                datetime.datetime(2021, 12, 30),
                "3mo14h15s11ms33us999ns",
                eager=True,
            )
        }
    )

    q = ldf.select(field(pl.col("datetimes").dt))

    assert_gpu_result_equal(q)


def test_datetime_extra_unsupported(monkeypatch):
    ldf = pl.LazyFrame(
        {
            "datetimes": pl.datetime_range(
                datetime.datetime(2020, 1, 1),
                datetime.datetime(2021, 12, 30),
                "3mo14h15s11ms33us999ns",
                eager=True,
            )
        }
    )

    def unsupported_name_setter(self, value):
        pass

    def unsupported_name_getter(self):
        return "unsupported"

    monkeypatch.setattr(
        TemporalFunction,
        "name",
        property(unsupported_name_getter, unsupported_name_setter),
    )

    q = ldf.select(pl.col("datetimes").dt.nanosecond())

    assert_ir_translation_raises(q, NotImplementedError)


@pytest.mark.parametrize(
    "field",
    [
        methodcaller("year"),
        methodcaller("month"),
        methodcaller("day"),
        methodcaller("weekday"),
    ],
)
def test_date_extract(field):
    ldf = pl.LazyFrame(
        {
            "dates": [
                datetime.date(2024, 1, 1),
                datetime.date(2024, 10, 11),
            ]
        }
    )

    ldf = pl.LazyFrame(
        {"dates": [datetime.date(2024, 1, 1), datetime.date(2024, 10, 11)]}
    )

    q = ldf.select(field(pl.col("dates").dt))

    assert_gpu_result_equal(q)
