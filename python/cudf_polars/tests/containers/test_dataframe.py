# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import cudf._lib.pylibcudf as plc

from cudf_polars.containers import DataFrame, NamedColumn


def test_select_missing_raises():
    df = DataFrame(
        [
            NamedColumn(
                plc.column_factories.make_numeric_column(
                    plc.DataType(plc.TypeId.INT8), 2, plc.MaskState.ALL_VALID
                ),
                "a",
            )
        ]
    )
    with pytest.raises(ValueError):
        df.select(["b", "a"])


def test_replace_missing_raises():
    df = DataFrame(
        [
            NamedColumn(
                plc.column_factories.make_numeric_column(
                    plc.DataType(plc.TypeId.INT8), 2, plc.MaskState.ALL_VALID
                ),
                "a",
            )
        ]
    )
    replacement = df.columns[0].copy(new_name="b")
    with pytest.raises(ValueError):
        df.replace_columns(replacement)


def test_from_table_wrong_names():
    table = plc.Table(
        [
            plc.column_factories.make_numeric_column(
                plc.DataType(plc.TypeId.INT8), 1, plc.MaskState.ALL_VALID
            )
        ]
    )
    with pytest.raises(ValueError):
        DataFrame.from_table(table, ["a", "b"])


def test_sorted_like_raises_mismatching_names():
    df = DataFrame(
        [
            NamedColumn(
                plc.column_factories.make_numeric_column(
                    plc.DataType(plc.TypeId.INT8), 2, plc.MaskState.ALL_VALID
                ),
                "a",
            )
        ]
    )
    like = df.copy().rename_columns({"a": "b"})
    with pytest.raises(ValueError):
        df.sorted_like(like)


def test_shallow_copy():
    column = NamedColumn(
        plc.column_factories.make_numeric_column(
            plc.DataType(plc.TypeId.INT8), 2, plc.MaskState.ALL_VALID
        ),
        "a",
    )
    column.set_sorted(
        is_sorted=plc.types.Sorted.YES,
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
    )
    df = DataFrame([column])
    copy = df.copy()
    copy.columns[0].set_sorted(
        is_sorted=plc.types.Sorted.NO,
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
    )
    assert df.columns[0].is_sorted == plc.types.Sorted.YES
    assert copy.columns[0].is_sorted == plc.types.Sorted.NO
