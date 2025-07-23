# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import DataType
from cudf_polars.utils.dtypes import is_order_preserving_cast

INT8 = plc.DataType(plc.TypeId.INT8)
INT16 = plc.DataType(plc.TypeId.INT16)
INT32 = plc.DataType(plc.TypeId.INT32)
INT64 = plc.DataType(plc.TypeId.INT64)
UINT8 = plc.DataType(plc.TypeId.UINT8)
UINT16 = plc.DataType(plc.TypeId.UINT16)
UINT32 = plc.DataType(plc.TypeId.UINT32)
UINT64 = plc.DataType(plc.TypeId.UINT64)
FLOAT32 = plc.DataType(plc.TypeId.FLOAT32)
FLOAT64 = plc.DataType(plc.TypeId.FLOAT64)


@pytest.mark.parametrize(
    "pltype",
    [
        pl.Time(),
        pl.Struct({"a": pl.Binary(), "b": pl.Float32}),
        pl.List(pl.Object()),
        pl.Array(pl.Int8, 2),
        pl.Binary(),
        pl.Categorical(),
        pl.Enum(["a", "b"]),
        pl.Field("a", pl.Int8),
        pl.Object(),
        pl.Unknown(),
    ],
    ids=repr,
)
def test_unhandled_dtype_conversion_raises(pltype):
    with pytest.raises(NotImplementedError):
        _ = DataType(pltype)


@pytest.mark.parametrize(
    "plc_dtype, polars_dtype",
    [
        (plc.DataType(plc.TypeId.BOOL8), pl.Boolean()),
        (plc.DataType(plc.TypeId.INT8), pl.Int8()),
        (plc.DataType(plc.TypeId.INT16), pl.Int16()),
        (plc.DataType(plc.TypeId.INT32), pl.Int32()),
        (plc.DataType(plc.TypeId.INT64), pl.Int64()),
        (plc.DataType(plc.TypeId.UINT8), pl.UInt8()),
        (plc.DataType(plc.TypeId.UINT16), pl.UInt16()),
        (plc.DataType(plc.TypeId.UINT32), pl.UInt32()),
        (plc.DataType(plc.TypeId.UINT64), pl.UInt64()),
        (plc.DataType(plc.TypeId.FLOAT32), pl.Float32()),
        (plc.DataType(plc.TypeId.FLOAT64), pl.Float64()),
        (plc.DataType(plc.TypeId.TIMESTAMP_DAYS), pl.Date()),
        (plc.DataType(plc.TypeId.EMPTY), pl.Null()),
        (plc.DataType(plc.TypeId.STRING), pl.String()),
        (plc.DataType(plc.TypeId.TIMESTAMP_MILLISECONDS), pl.Datetime("ms")),
        (plc.DataType(plc.TypeId.TIMESTAMP_MICROSECONDS), pl.Datetime("us")),
        (plc.DataType(plc.TypeId.TIMESTAMP_NANOSECONDS), pl.Datetime("ns")),
        (plc.DataType(plc.TypeId.DURATION_MILLISECONDS), pl.Duration("ms")),
        (plc.DataType(plc.TypeId.DURATION_MICROSECONDS), pl.Duration("us")),
        (plc.DataType(plc.TypeId.DURATION_NANOSECONDS), pl.Duration("ns")),
    ],
    ids=lambda d: f"{d!r}",
)
def test_plc_to_polars_dtype(plc_dtype, polars_dtype):
    dtype = DataType(plc_dtype)
    assert dtype.polars == polars_dtype
    assert dtype.plc == plc_dtype


@pytest.mark.parametrize(
    "plc_dtype",
    [
        plc.DataType(plc.TypeId.LIST),
        plc.DataType(plc.TypeId.STRUCT),
        plc.DataType(plc.TypeId.DICTIONARY32),
        plc.DataType(plc.TypeId.DECIMAL32),
        plc.DataType(plc.TypeId.DECIMAL64),
        plc.DataType(plc.TypeId.DECIMAL128),
    ],
    ids=lambda d: f"{d.id().name}",
)
def test_unhandled_plc_dtype_conversion_raises(plc_dtype):
    with pytest.raises(NotImplementedError):
        _ = DataType(plc_dtype)


def test_is_order_preserving_cast():
    assert is_order_preserving_cast(INT8, INT8)  # Same type
    assert is_order_preserving_cast(INT8, INT16)  # Smaller type
    assert is_order_preserving_cast(INT8, FLOAT32)  # Int to large enough float
    assert is_order_preserving_cast(UINT8, UINT16)  # Unsigned to larger unsigned
    assert is_order_preserving_cast(UINT8, FLOAT32)  # Unsigned to large enough float
    assert is_order_preserving_cast(FLOAT32, FLOAT64)  # Float to larger float
    assert is_order_preserving_cast(INT64, FLOAT32)  # Int any float
    assert is_order_preserving_cast(FLOAT32, INT32)  # Float to undersized int
    assert is_order_preserving_cast(FLOAT32, INT64)  # float to large int

    assert not is_order_preserving_cast(INT16, INT8)  # Bigger type
    assert not is_order_preserving_cast(INT8, UINT8)  # Different signedness
    assert not is_order_preserving_cast(FLOAT64, FLOAT32)  # Smaller float
