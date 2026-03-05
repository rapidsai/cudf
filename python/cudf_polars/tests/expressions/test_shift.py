# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("n", [0, 1, 2, -1, -2, 5, -5])
def test_shift(n):
    df = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    q = df.select(pl.col("a").shift(n))
    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "n,fill_value",
    [
        (1, 0),
        (2, 99),
        (-1, 0),
        (-2, -1),
        (0, 7),
    ],
)
def test_shift_and_fill(n, fill_value):
    df = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    q = df.select(pl.col("a").shift(n, fill_value=fill_value))
    assert_gpu_result_equal(q)


def test_shift_multiple_columns():
    df = pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    q = df.select(pl.col("a").shift(2), pl.col("b").shift(-1))
    assert_gpu_result_equal(q)


def test_shift_float():
    df = pl.LazyFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    q = df.select(pl.col("a").shift(2))
    assert_gpu_result_equal(q)


def test_shift_and_fill_float():
    df = pl.LazyFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    q = df.select(pl.col("a").shift(2, fill_value=0.0))
    assert_gpu_result_equal(q)


def test_shift_string():
    df = pl.LazyFrame({"a": ["x", "y", "z"]})
    q = df.select(pl.col("a").shift(1))
    assert_gpu_result_equal(q)


def test_shift_with_columns():
    df = pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
    q = df.with_columns(shift=pl.col("a").shift(-2, fill_value=100))
    assert_gpu_result_equal(q)


@pytest.mark.parametrize("n", [1, -1, 2])
def test_shift_datetime(n):
    from datetime import datetime

    df = pl.LazyFrame(
        {
            "a": [datetime(2021, 1, 1), datetime(2021, 1, 2), datetime(2021, 1, 3)],
            "b": [datetime(2022, 6, 1), datetime(2022, 6, 2), datetime(2022, 6, 3)],
        }
    )
    q = df.select(pl.col("a").shift(n), pl.col("b").shift(-n))
    assert_gpu_result_equal(q)


@pytest.mark.parametrize("n", [1, -1])
def test_shift_date(n):
    from datetime import date

    df = pl.LazyFrame({"a": [date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3)]})
    q = df.select(pl.col("a").shift(n))
    assert_gpu_result_equal(q)
