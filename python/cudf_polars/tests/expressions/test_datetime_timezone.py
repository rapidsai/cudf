# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import datetime
from typing import Literal, cast

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)

UNITS = ["ms", "us", "ns"]


_SAMPLE_DATETIMES = [
    datetime.datetime(2020, 1, 15, 12, 0, 0),
    datetime.datetime(2020, 6, 15, 12, 30, 15),
    datetime.datetime(2019, 11, 15, 12, 45, 30),
    datetime.datetime(1969, 6, 15, 12, 0, 0),
    None,
]


def _datetime(unit: str) -> pl.Datetime:
    return pl.Datetime(cast("Literal['ms', 'us', 'ns']", unit))


def _utc_frame(unit: str) -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "a": pl.Series(
                _SAMPLE_DATETIMES, dtype=_datetime(unit)
            ).dt.replace_time_zone("UTC")
        }
    )


def _naive_frame(unit: str) -> pl.LazyFrame:
    return pl.LazyFrame({"a": pl.Series(_SAMPLE_DATETIMES, dtype=_datetime(unit))})


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize(
    "target", ["Europe/London", "US/Pacific", "Asia/Kolkata", "UTC", "Etc/GMT"]
)
def test_convert_time_zone_from_utc(engine, unit, target):
    q = _utc_frame(unit).select(pl.col("a").dt.convert_time_zone(target))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("unit", UNITS)
def test_convert_time_zone_from_naive(engine, unit):
    q = _naive_frame(unit).select(pl.col("a").dt.convert_time_zone("Europe/London"))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("unit", UNITS)
def test_convert_time_zone_between_zones(engine, unit):
    q = (
        _utc_frame(unit)
        .with_columns(pl.col("a").dt.convert_time_zone("Europe/London"))
        .select(pl.col("a").dt.convert_time_zone("US/Pacific"))
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize(
    "target",
    ["Europe/Amsterdam", "US/Pacific", "Asia/Kolkata", "Etc/GMT-5", "UTC", None],
)
def test_replace_time_zone_from_naive(engine, unit, target):
    q = _naive_frame(unit).select(pl.col("a").dt.replace_time_zone(target))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("unit", UNITS)
@pytest.mark.parametrize("target", ["Europe/Amsterdam", "US/Pacific", "UTC", None])
def test_replace_time_zone_from_aware(engine, unit, target):
    q = (
        _utc_frame(unit)
        .with_columns(pl.col("a").dt.convert_time_zone("Europe/London"))
        .select(pl.col("a").dt.replace_time_zone(target))
    )
    assert_gpu_result_equal(q, engine=engine)


def test_replace_time_zone_from_empty_table_zone(engine):
    q = (
        _utc_frame("us")
        .with_columns(pl.col("a").dt.convert_time_zone("Etc/GMT"))
        .select(pl.col("a").dt.replace_time_zone("Etc/GMT"))
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("ambiguous", ["earliest", "latest", "null"])
def test_replace_time_zone_ambiguous(engine, ambiguous):
    q = pl.LazyFrame(
        {
            "a": [
                datetime.datetime(2018, 10, 28, 2, 30),
                datetime.datetime(2020, 1, 1),
                None,
            ]
        }
    ).select(pl.col("a").dt.replace_time_zone("Europe/Amsterdam", ambiguous=ambiguous))
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("non_existent", ["null"])
def test_replace_time_zone_non_existent_null(engine, non_existent):
    q = pl.LazyFrame(
        {
            "a": [
                datetime.datetime(2018, 3, 25, 2, 30),
                datetime.datetime(2020, 1, 1),
                None,
            ]
        }
    ).select(
        pl.col("a").dt.replace_time_zone("Europe/Amsterdam", non_existent=non_existent)
    )
    assert_gpu_result_equal(q, engine=engine)


def test_replace_time_zone_ambiguous_per_row(engine):
    q = pl.LazyFrame(
        {
            "a": [
                datetime.datetime(2018, 10, 28, 2, 30),
                datetime.datetime(2018, 10, 28, 2, 30),
                datetime.datetime(2018, 10, 28, 2, 30),
                datetime.datetime(2020, 1, 1),
            ],
            "ambiguous": ["earliest", "latest", "null", "raise"],
        }
    ).select(
        pl.col("a").dt.replace_time_zone(
            "Europe/Amsterdam", ambiguous=pl.col("ambiguous")
        )
    )
    assert_gpu_result_equal(q, engine=engine)


def test_replace_time_zone_same_zone_ambiguous_instant(engine):
    q = (
        pl.LazyFrame({"a": [datetime.datetime(2018, 10, 28, 2, 30)]})
        .with_columns(
            pl.col("a").dt.replace_time_zone("Europe/Amsterdam", ambiguous="earliest")
        )
        .select(pl.col("a").dt.replace_time_zone("Europe/Amsterdam"))
    )
    assert_gpu_result_equal(q, engine=engine)


def test_replace_time_zone_ambiguous_raises(engine_raise_on_fail):
    q = pl.LazyFrame({"a": [datetime.datetime(2018, 10, 28, 2, 30)]}).select(
        pl.col("a").dt.replace_time_zone("Europe/Amsterdam")
    )
    with pytest.raises(pl.exceptions.ComputeError):
        q.collect(engine=engine_raise_on_fail)


def test_replace_time_zone_non_existent_raises(engine_raise_on_fail):
    q = pl.LazyFrame({"a": [datetime.datetime(2018, 3, 25, 2, 30)]}).select(
        pl.col("a").dt.replace_time_zone("Europe/Amsterdam")
    )
    with pytest.raises(pl.exceptions.ComputeError):
        q.collect(engine=engine_raise_on_fail)


def test_replace_time_zone_ambiguous_per_row_raises(engine_raise_on_fail):
    q = pl.LazyFrame(
        {
            "a": [
                datetime.datetime(2018, 10, 28, 2, 30),
                datetime.datetime(2020, 1, 1),
            ],
            "ambiguous": ["raise", "raise"],
        }
    ).select(
        pl.col("a").dt.replace_time_zone(
            "Europe/Amsterdam", ambiguous=pl.col("ambiguous")
        )
    )
    with pytest.raises(pl.exceptions.ComputeError):
        q.collect(engine=engine_raise_on_fail)


def test_replace_time_zone_unknown_zone_raises(engine, monkeypatch):
    import cudf_polars.dsl.expressions.datetime as dtmod

    monkeypatch.setattr(dtmod.zoneinfo, "TZPATH", ())
    q = _naive_frame("us").select(pl.col("a").dt.replace_time_zone("Europe/Amsterdam"))
    assert_ir_translation_raises(q, engine, NotImplementedError)
