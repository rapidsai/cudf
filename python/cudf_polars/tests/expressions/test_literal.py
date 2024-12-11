# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils import dtypes


@pytest.fixture(
    params=[
        None,
        pl.Int8(),
        pl.Int16(),
        pl.Int32(),
        pl.Int64(),
        pl.UInt8(),
        pl.UInt16(),
        pl.UInt32(),
        pl.UInt64(),
    ]
)
def integer(request):
    return pl.lit(10, dtype=request.param)


@pytest.fixture(params=[None, pl.Float32(), pl.Float64()])
def float(request):
    return pl.lit(1.0, dtype=request.param)


def test_numeric_literal(integer, float):
    df = pl.LazyFrame({})

    q = df.select(integer=integer, float_=float, sum_=integer + float)

    assert_gpu_result_equal(q)


@pytest.fixture(
    params=[pl.Date(), pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]
)
def timestamp(request):
    return pl.lit(10_000, dtype=request.param)


@pytest.fixture(params=[pl.Duration("ms"), pl.Duration("us"), pl.Duration("ns")])
def timedelta(request):
    return pl.lit(9_000, dtype=request.param)


def test_timelike_literal(timestamp, timedelta):
    df = pl.LazyFrame({})

    q = df.select(
        time=timestamp,
        delta=timedelta,
        adjusted=timestamp + timedelta,
        two_delta=timedelta + timedelta,
    )
    schema = {k: dtypes.from_polars(v) for k, v in q.collect_schema().items()}
    if plc.binaryop.is_supported_operation(
        schema["adjusted"],
        schema["time"],
        schema["delta"],
        plc.binaryop.BinaryOperator.ADD,
    ) and plc.binaryop.is_supported_operation(
        schema["two_delta"],
        schema["delta"],
        schema["delta"],
        plc.binaryop.BinaryOperator.ADD,
    ):
        assert_gpu_result_equal(q)
    else:
        assert_ir_translation_raises(q, NotImplementedError)


def test_select_literal_series():
    df = pl.LazyFrame({})

    q = df.select(
        a=pl.Series(["a", "b", "c"], dtype=pl.String()),
        b=pl.Series([[1, 2], [3], None], dtype=pl.List(pl.UInt16())),
        c=pl.Series([[[1]], [], [[1, 2, 3, 4]]], dtype=pl.List(pl.List(pl.Float32()))),
    )

    assert_gpu_result_equal(q)


@pytest.mark.parametrize("expr", [pl.lit(None), pl.lit(10, dtype=pl.Decimal())])
def test_unsupported_literal_raises(expr):
    df = pl.LazyFrame({})

    q = df.select(expr)

    assert_ir_translation_raises(q, NotImplementedError)
