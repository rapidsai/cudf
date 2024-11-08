# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pyarrow as pa
import pytest

import polars as pl
from polars.testing.asserts import assert_frame_equal

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame
from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_select_missing_raises():
    df = DataFrame(
        [
            Column(
                plc.column_factories.make_numeric_column(
                    plc.DataType(plc.TypeId.INT8), 2, plc.MaskState.ALL_VALID
                ),
                name="a",
            )
        ]
    )
    with pytest.raises(ValueError):
        df.select(["b", "a"])


def test_replace_missing_raises():
    df = DataFrame(
        [
            Column(
                plc.column_factories.make_numeric_column(
                    plc.DataType(plc.TypeId.INT8), 2, plc.MaskState.ALL_VALID
                ),
                name="a",
            )
        ]
    )
    replacement = df.column_map["a"].copy().rename("b")
    with pytest.raises(ValueError):
        df.with_columns([replacement], replace_only=True)


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


def test_unnamed_column_raise():
    payload = plc.column_factories.make_numeric_column(
        plc.DataType(plc.TypeId.INT8), 0, plc.MaskState.ALL_VALID
    )

    with pytest.raises(ValueError):
        DataFrame([Column(payload, name="a"), Column(payload)])


def test_sorted_like_raises_mismatching_names():
    df = DataFrame(
        [
            Column(
                plc.column_factories.make_numeric_column(
                    plc.DataType(plc.TypeId.INT8), 2, plc.MaskState.ALL_VALID
                ),
                name="a",
            )
        ]
    )
    like = df.copy().rename_columns({"a": "b"})
    with pytest.raises(ValueError):
        df.sorted_like(like)


def test_shallow_copy():
    column = Column(
        plc.column_factories.make_numeric_column(
            plc.DataType(plc.TypeId.INT8), 2, plc.MaskState.ALL_VALID
        ),
        name="a",
    )
    column.set_sorted(
        is_sorted=plc.types.Sorted.YES,
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
    )
    df = DataFrame([column])
    copy = df.copy()
    copy.column_map["a"].set_sorted(
        is_sorted=plc.types.Sorted.NO,
        order=plc.types.Order.ASCENDING,
        null_order=plc.types.NullOrder.AFTER,
    )
    assert df.column_map["a"].is_sorted == plc.types.Sorted.YES
    assert copy.column_map["a"].is_sorted == plc.types.Sorted.NO


def test_sorted_flags_preserved_empty():
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int8())})
    df.select(pl.col("a").sort())

    gf = DataFrame.from_polars(df)

    a = gf.column_map["a"]

    assert a.is_sorted == plc.types.Sorted.YES

    assert df.flags == gf.to_polars().flags


@pytest.mark.parametrize("nulls_last", [True, False])
def test_sorted_flags_preserved(with_nulls, nulls_last):
    values = [1, 2, -1, 2, 4, 5]
    if with_nulls:
        values[4] = None
    df = pl.DataFrame({"a": values, "b": values, "c": values})

    df = df.select(
        pl.col("a").sort(descending=False, nulls_last=nulls_last),
        pl.col("b").sort(descending=True, nulls_last=nulls_last),
        pl.col("c"),
    )

    gf = DataFrame.from_polars(df)

    a_null_order = (
        plc.types.NullOrder.AFTER
        if nulls_last and with_nulls
        else plc.types.NullOrder.BEFORE
    )
    b_null_order = (
        plc.types.NullOrder.AFTER
        if not nulls_last and with_nulls
        else plc.types.NullOrder.BEFORE
    )
    a, b, c = gf.columns
    assert a.is_sorted == plc.types.Sorted.YES
    assert a.order == plc.types.Order.ASCENDING
    assert a.null_order == a_null_order
    assert b.is_sorted == plc.types.Sorted.YES
    assert b.order == plc.types.Order.DESCENDING
    assert b.null_order == b_null_order
    assert c.is_sorted == plc.types.Sorted.NO
    assert df.flags == gf.to_polars().flags


def test_empty_name_roundtrips_overlap():
    df = pl.LazyFrame({"": [1, 2, 3], "column_0": [4, 5, 6]})
    assert_gpu_result_equal(df)


def test_empty_name_roundtrips_no_overlap():
    df = pl.LazyFrame({"": [1, 2, 3], "b": [4, 5, 6]})
    assert_gpu_result_equal(df)


@pytest.mark.parametrize(
    "arrow_tbl",
    [
        pa.table([]),
        pa.table({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}),
        pa.table({"a": [1, 2, 3]}),
        pa.table({"a": [1], "b": [2], "c": [3]}),
        pa.table({"a": ["a", "bb", "ccc"]}),
        pa.table({"a": [1, 2, None], "b": [None, 3, 4]}),
    ],
)
def test_serialization_roundtrip(arrow_tbl):
    plc_tbl = plc.interop.from_arrow(arrow_tbl)
    df = DataFrame.from_table(plc_tbl, names=arrow_tbl.column_names)

    header, frames = df.serialize()
    res = DataFrame.deserialize(header, frames)

    assert_frame_equal(df.to_polars(), res.to_polars())
