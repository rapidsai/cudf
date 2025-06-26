# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

import pylibcudf as plc

import cudf_polars.containers.datatype
from cudf_polars.containers import Column, DataType


def test_non_scalar_access_raises():
    dtype = DataType(pl.Int8())
    column = Column(
        plc.column_factories.make_numeric_column(dtype.plc, 2, plc.MaskState.ALL_VALID),
        dtype=dtype,
    )
    with pytest.raises(ValueError):
        _ = column.obj_scalar


@pytest.mark.parametrize("length", [0, 1])
def test_length_leq_one_always_sorted(length):
    dtype = DataType(pl.Int8())
    column = Column(
        plc.column_factories.make_numeric_column(
            dtype.plc, length, plc.MaskState.ALL_VALID
        ),
        dtype=dtype,
    )
    assert column.check_sorted(
        order=plc.types.Order.ASCENDING, null_order=plc.types.NullOrder.AFTER
    )
    assert column.check_sorted(
        order=plc.types.Order.DESCENDING, null_order=plc.types.NullOrder.AFTER
    )

    column.set_sorted(
        is_sorted=plc.types.Sorted.NO,
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
    )
    assert column.check_sorted(
        order=plc.types.Order.ASCENDING, null_order=plc.types.NullOrder.AFTER
    )
    assert column.check_sorted(
        order=plc.types.Order.DESCENDING, null_order=plc.types.NullOrder.AFTER
    )


def test_shallow_copy():
    dtype = DataType(pl.Int8())
    column = Column(
        plc.column_factories.make_numeric_column(dtype.plc, 2, plc.MaskState.ALL_VALID),
        dtype=dtype,
    )
    copy = column.copy()
    copy = copy.set_sorted(
        is_sorted=plc.types.Sorted.YES,
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
    )
    assert column.is_sorted == plc.types.Sorted.NO
    assert copy.is_sorted == plc.types.Sorted.YES


@pytest.mark.parametrize("typeid", [pl.Int8(), pl.Float32()])
def test_mask_nans(typeid):
    dtype = DataType(typeid)
    column = Column(
        plc.Column.from_iterable_of_py([0, 0, 0], dtype=dtype.plc), dtype=dtype
    )
    masked = column.mask_nans()
    assert column.null_count == masked.null_count


def test_mask_nans_float():
    dtype = DataType(pl.Float32())
    column = Column(
        plc.Column.from_iterable_of_py([0, 0, float("nan")], dtype=dtype.plc),
        dtype=dtype,
    )
    masked = column.mask_nans()
    assert masked.nan_count == 0
    assert masked.slice((0, 2)).null_count == 0
    assert masked.slice((2, 1)).null_count == 1


def test_slice_none_returns_self():
    dtype = DataType(pl.Int8())
    column = Column(
        plc.column_factories.make_numeric_column(dtype.plc, 2, plc.MaskState.ALL_VALID),
        dtype=dtype,
    )
    assert column.slice(None) is column


def test_deserialize_ctor_kwargs_invalid_dtype():
    column_kwargs = {
        "is_sorted": plc.types.Sorted.NO,
        "order": plc.types.Order.ASCENDING,
        "null_order": plc.types.NullOrder.AFTER,
        "name": "test",
        "dtype": "in64",
    }
    with pytest.raises(ValueError):
        Column.deserialize_ctor_kwargs(column_kwargs)


def test_deserialize_ctor_kwargs_list_dtype():
    pl_type = pl.List(pl.Int64())
    column_kwargs = {
        "is_sorted": plc.types.Sorted.NO,
        "order": plc.types.Order.ASCENDING,
        "null_order": plc.types.NullOrder.AFTER,
        "name": "test",
        "dtype": pl.polars.dtype_str_repr(pl_type),
    }
    result = Column.deserialize_ctor_kwargs(column_kwargs)
    expected = {
        "is_sorted": plc.types.Sorted.NO,
        "order": plc.types.Order.ASCENDING,
        "null_order": plc.types.NullOrder.AFTER,
        "name": "test",
        "dtype": DataType(pl_type),
    }
    assert result == expected


def test_serialize_cache_miss():
    dtype = DataType(pl.Int8())
    column = Column(
        plc.column_factories.make_numeric_column(dtype.plc, 2, plc.MaskState.ALL_VALID),
        dtype=dtype,
    )
    header, frames = column.serialize()
    assert header == {"column_kwargs": column.serialize_ctor_kwargs(), "frame_count": 2}
    assert len(frames) == 2
    assert frames[0].nbytes > 0
    assert frames[1].nbytes > 0

    # https://github.com/rapidsai/cudf/pull/18953
    # In a multi-GPU setup, we might attempt to deserialize a column
    # whose type we haven't seen before. polars lets you use either the
    # class (`pl.Int8`) or an instance (`pl.Int8()`) in most places
    # for types that are not parameterized. These are equal and have
    # the same hash, so they cache the same, but have some difference
    # in behavior (e.g. isinstance).
    cudf_polars.containers.datatype._from_polars.cache_clear()
    result = Column.deserialize(header, frames)
    assert result.dtype == dtype
