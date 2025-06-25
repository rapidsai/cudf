# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pyarrow
import pytest

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataType


def test_non_scalar_access_raises():
    dtype = DataType(pl.Int8())
    column = Column(
        plc.column_factories.make_numeric_column(dtype.plc, 2, plc.MaskState.ALL_VALID),
        dtype=dtype,
    )
    with pytest.raises(ValueError):
        _ = column.obj_scalar


def test_check_sorted():
    dtype = DataType(pl.Int8())
    column = Column(
        plc.Column.from_iterable_of_py([0, 1, 2], dtype.plc),
        dtype=dtype,
    )
    assert column.check_sorted(
        order=plc.types.Order.ASCENDING, null_order=plc.types.NullOrder.AFTER
    )
    column.set_sorted(
        is_sorted=plc.types.Sorted.YES,
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
    )
    assert column.check_sorted(
        order=plc.types.Order.ASCENDING, null_order=plc.types.NullOrder.AFTER
    )


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
    values = pyarrow.array([0, 0, 0], type=plc.interop.to_arrow(dtype.plc))
    column = Column(plc.Column.from_arrow(values), dtype=dtype)
    masked = column.mask_nans()
    assert column.null_count == masked.null_count


def test_mask_nans_float():
    dtype = DataType(pl.Float32())
    values = pyarrow.array([0, 0, float("nan")], type=plc.interop.to_arrow(dtype.plc))
    column = Column(plc.Column.from_arrow(values), dtype=dtype)
    masked = column.mask_nans()
    expect = pyarrow.array([0, 0, None], type=plc.interop.to_arrow(dtype.plc))
    got = pyarrow.array(plc.interop.to_arrow(masked.obj))

    assert expect == got


def test_slice_none_returns_self():
    dtype = DataType(pl.Int8())
    column = Column(
        plc.column_factories.make_numeric_column(dtype.plc, 2, plc.MaskState.ALL_VALID),
        dtype=dtype,
    )
    assert column.slice(None) is column
