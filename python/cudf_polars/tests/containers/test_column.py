# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import cudf._lib.pylibcudf as plc

from cudf_polars.containers import Column


def test_non_scalar_access_raises():
    column = Column(
        plc.column_factories.make_numeric_column(
            plc.DataType(plc.TypeId.INT8), 2, plc.MaskState.ALL_VALID
        )
    )
    with pytest.raises(ValueError):
        _ = column.obj_scalar


def test_length_one_always_sorted():
    column = Column(
        plc.column_factories.make_numeric_column(
            plc.DataType(plc.TypeId.INT8), 1, plc.MaskState.ALL_VALID
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
