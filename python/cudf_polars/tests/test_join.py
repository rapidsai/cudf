# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from decimal import Decimal

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
    get_default_engine,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_132


@pytest.fixture(params=[False, True], ids=["nulls_not_equal", "nulls_equal"])
def nulls_equal(request):
    return request.param


@pytest.fixture(params=["inner", "left", "right", "semi", "anti", "full"])
def how(request):
    return request.param


@pytest.fixture(params=[None, (1, 5), (1, None), (0, 2), (0, None)])
def zlice(request):
    return request.param


@pytest.fixture
def left():
    return pl.LazyFrame(
        {
            "a": [1, 2, 3, 1, None],
            "b": [1, 2, 3, 4, 5],
            "c": [2, 3, 4, 5, 6],
        }
    )


@pytest.fixture
def right():
    return pl.LazyFrame(
        {
            "a": [1, 4, 3, 7, None, None, 1],
            "c": [2, 3, 4, 5, 6, 7, 8],
            "d": [6, None, 7, 8, -1, 2, 4],
        }
    )


@pytest.mark.parametrize(
    "maintain_order", ["left", "left_right", "right_left", "right"]
)
def test_join_maintain_order_param_unsupported(left, right, maintain_order):
    q = left.join(right, on=pl.col("a"), how="inner", maintain_order=maintain_order)

    assert_ir_translation_raises(q, NotImplementedError)


@pytest.mark.parametrize(
    "join_expr",
    [
        pl.col("a"),
        pl.col("a") * 2,
        [pl.col("a"), pl.col("c") + 1],
        ["c", "a"],
    ],
)
def test_non_coalesce_join(left, right, how, nulls_equal, join_expr):
    query = left.join(
        right, on=join_expr, how=how, nulls_equal=nulls_equal, coalesce=False
    )
    assert_gpu_result_equal(query, check_row_order=False)


@pytest.mark.parametrize(
    "join_expr",
    [
        pl.col("a"),
        ["c", "a"],
    ],
)
def test_coalesce_join(left, right, how, nulls_equal, join_expr):
    query = left.join(
        right, on=join_expr, how=how, nulls_equal=nulls_equal, coalesce=True
    )
    assert_gpu_result_equal(query, check_row_order=False)


def test_left_join_with_slice(left, right, nulls_equal, zlice):
    q = left.join(right, on="a", how="left", nulls_equal=nulls_equal, coalesce=True)

    if zlice is not None:
        # neither polars nor cudf guarantee row order.
        # left.join(right).slice(slice) is a fundamentally sensitive to
        # the row ordering of the join algorithm. So we can just check
        # the things that invariant to the ordering.
        q = q.slice(*zlice)

        engine = get_default_engine()

        # Check the number of rows
        assert_gpu_result_equal(q.select(pl.len()))

        # Check that the schema matches
        result = q.collect(engine=engine)
        assert result.schema == q.collect_schema()

    else:
        assert_gpu_result_equal(q, check_row_order=False)


def test_cross_join(left, right, zlice):
    q = left.join(right, how="cross")
    if zlice is not None:
        q = q.slice(*zlice)

    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "left_on,right_on",
    [
        (pl.col("a"), pl.lit(2, dtype=pl.Int64)),
        (pl.lit(2, dtype=pl.Int64), pl.col("a")),
    ],
)
def test_join_literal_key(left, right, left_on, right_on):
    q = left.join(right, left_on=left_on, right_on=right_on, how="inner")
    assert_gpu_result_equal(q, check_row_order=False)


@pytest.mark.parametrize(
    "conditions",
    [
        [pl.col("a") < pl.col("a_right")],
        [
            pl.col("a_right") <= pl.col("a") * 2,
            pl.col("a_right") <= 2 * pl.col("a"),
        ],
        [pl.col("b") * 2 > pl.col("a_right"), pl.col("a") == pl.col("c_right")],
        [pl.col("b") * 2 <= pl.col("a_right"), pl.col("a") < pl.col("c_right")],
        [pl.col("b") <= pl.col("a_right") * 7, pl.col("a") < pl.col("d") * 2],
    ],
)
@pytest.mark.parametrize("zlice", [None, (0, 5)])
def test_join_where(left, right, conditions, zlice):
    q = left.join_where(right, *conditions)

    assert_gpu_result_equal(q, check_row_order=False)

    if zlice is not None:
        q_len = q.slice(*zlice).select(pl.len())
        # Can't compare result, since row order is not guaranteed and
        # therefore we only check the length

        assert_gpu_result_equal(q_len)


def test_cross_join_empty_right_table(request):
    request.applymarker(
        pytest.mark.xfail(condition=POLARS_VERSION_LT_132, reason="nested loop join")
    )
    a = pl.LazyFrame({"a": [1, 2, 3], "x": [7, 2, 1]})
    b = pl.LazyFrame({"b": [2, 2, 2], "x": [7, 1, 3]})

    q = a.join(b, how="cross").filter(
        (pl.col("a") == pl.col("a")) & (pl.col("b") < pl.col("b"))
    )

    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("foo") > pl.col("bar"),
        pl.col("foo") >= pl.col("bar"),
        pl.col("foo") < pl.col("bar"),
        pl.col("foo") <= pl.col("bar"),
        pl.col("foo") == pl.col("bar"),
        pytest.param(
            pl.col("foo") != pl.col("bar"),
            marks=pytest.mark.xfail(reason="nested loop join"),
        ),
    ],
)
@pytest.mark.parametrize(
    "left_dtype,right_dtype",
    [
        (pl.Decimal(15, 2), pl.Decimal(15, 2)),
        (pl.Decimal(15, 4), pl.Decimal(15, 2)),
        (pl.Decimal(15, 2), pl.Decimal(15, 4)),
        (pl.Decimal(15, 2), pl.Float32),
        (pl.Decimal(15, 2), pl.Float64),
    ],
)
def test_cross_join_filter_with_decimals(expr, left_dtype, right_dtype):
    left = pl.LazyFrame(
        {
            "foo": [Decimal("1.00"), Decimal("2.50"), Decimal("3.00")],
            "foo1": [10, 20, 30],
        },
        schema={"foo": left_dtype, "foo1": pl.Int64},
    )

    if isinstance(right_dtype, pl.Decimal):
        right = pl.LazyFrame(
            {
                "bar": [Decimal("2").scaleb(-right_dtype.scale)],
                "foo1": ["x"],
            },
            schema={"bar": right_dtype, "foo1": pl.String},
        )
    else:
        right = pl.LazyFrame(
            {"bar": [2.0], "foo1": ["x"]},
            schema={"bar": right_dtype, "foo1": pl.String},
        )

    q = left.join(right, how="cross").filter(expr)

    assert_gpu_result_equal(q, check_row_order=False)
