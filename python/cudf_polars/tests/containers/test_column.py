# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pyarrow
import pytest

import pylibcudf as plc

from cudf_polars.containers import Column


def test_non_scalar_access_raises():
    column = Column(
        plc.column_factories.make_numeric_column(
            plc.DataType(plc.TypeId.INT8), 2, plc.MaskState.ALL_VALID
        )
    )
    with pytest.raises(ValueError):
        _ = column.obj_scalar


@pytest.mark.parametrize("length", [0, 1])
def test_length_leq_one_always_sorted(length):
    column = Column(
        plc.column_factories.make_numeric_column(
            plc.DataType(plc.TypeId.INT8), length, plc.MaskState.ALL_VALID
        )
    )
    assert column.is_sorted == plc.types.Sorted.YES
    column.set_sorted(
        is_sorted=plc.types.Sorted.NO,
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
    )
    assert column.is_sorted == plc.types.Sorted.YES


def test_shallow_copy():
    column = Column(
        plc.column_factories.make_numeric_column(
            plc.DataType(plc.TypeId.INT8), 2, plc.MaskState.ALL_VALID
        )
    )
    copy = column.copy()
    copy = copy.set_sorted(
        is_sorted=plc.types.Sorted.YES,
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
    )
    assert column.is_sorted == plc.types.Sorted.NO
    assert copy.is_sorted == plc.types.Sorted.YES


@pytest.mark.parametrize("typeid", [plc.TypeId.INT8, plc.TypeId.FLOAT32])
def test_mask_nans(typeid):
    dtype = plc.DataType(typeid)
    values = pyarrow.array([0, 0, 0], type=plc.interop.to_arrow(dtype))
    column = Column(plc.interop.from_arrow(values))
    masked = column.mask_nans()
    assert column.obj.null_count() == masked.obj.null_count()


def test_mask_nans_float():
    dtype = plc.DataType(plc.TypeId.FLOAT32)
    values = pyarrow.array([0, 0, float("nan")], type=plc.interop.to_arrow(dtype))
    column = Column(plc.interop.from_arrow(values))
    masked = column.mask_nans()
    expect = pyarrow.array([0, 0, None], type=plc.interop.to_arrow(dtype))
    got = pyarrow.array(plc.interop.to_arrow(masked.obj))

    assert expect == got
