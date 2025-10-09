# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from decimal import Decimal

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    get_default_engine,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_130, POLARS_VERSION_LT_132


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
def test_join_maintain_order(left, right, maintain_order):
    q = left.join(right, on=pl.col("a"), how="inner", maintain_order=maintain_order)

    assert_gpu_result_equal(q)


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


@pytest.mark.parametrize("maintain_order", ["left_right", "right_left"])
@pytest.mark.parametrize("how", ["inner", "full"])
def test_join_maintain_order_inner_full(left, right, how, maintain_order, nulls_equal):
    q = left.join(
        right, on="a", how=how, nulls_equal=nulls_equal, maintain_order=maintain_order
    )
    assert_gpu_result_equal(q)


@pytest.mark.parametrize("maintain_order", ["left", "left_right"])
def test_join_maintain_order_left(left, right, maintain_order, nulls_equal):
    q = left.join(
        right,
        on="a",
        how="left",
        nulls_equal=nulls_equal,
        maintain_order=maintain_order,
    )
    assert_gpu_result_equal(q)


@pytest.mark.parametrize("maintain_order", ["right", "right_left"])
def test_join_maintain_order_right(left, right, maintain_order, nulls_equal):
    q = left.join(
        right,
        on="a",
        how="right",
        nulls_equal=nulls_equal,
        maintain_order=maintain_order,
    )
    assert_gpu_result_equal(q)


@pytest.mark.parametrize("maintain_order", ["left_right", "right_left"])
@pytest.mark.parametrize("join_expr", [pl.col("a"), ["c", "a"]])
@pytest.mark.parametrize("how", ["inner", "full"])
def test_join_maintain_order_multiple_keys(left, right, how, join_expr, maintain_order):
    q = left.join(right, on=join_expr, how=how, maintain_order=maintain_order)
    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "maintain_order,how",
    [
        ("left", "left"),
        ("left_right", "left"),
        ("right", "right"),
        ("right_left", "right"),
        ("left_right", "full"),
        ("right_left", "full"),
    ],
)
def test_join_maintain_order_with_coalesce(left, right, maintain_order, how):
    q = left.join(right, on="a", how=how, coalesce=True, maintain_order=maintain_order)
    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "maintain_order,how,zlice",
    [
        ("left_right", "inner", (0, 3)),
        ("right_left", "inner", (1, 3)),
        ("left_right", "full", (0, 4)),
        ("right_left", "full", (2, 3)),
    ],
)
def test_join_maintain_order_with_slice(left, right, maintain_order, how, zlice):
    # Need to disable slice pushdown to make the test deterministic. We want to materialize
    # the full join result and then slice
    q = left.join(right, on="a", how=how, maintain_order=maintain_order).slice(*zlice)
    assert_gpu_result_equal(
        q,
        polars_collect_kwargs={"slice_pushdown": False}
        if POLARS_VERSION_LT_130
        else {"optimizations": pl.QueryOptFlags(slice_pushdown=False)},
    )


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
def test_cross_join_filter_with_decimals(request, expr, left_dtype, right_dtype):
    request.applymarker(
        pytest.mark.xfail(
            POLARS_VERSION_LT_132
            and isinstance(left_dtype, pl.Decimal)
            and isinstance(right_dtype, pl.Decimal)
            and "==" in repr(expr),
            reason="Hash Inner Join between i128 and i128",
        )
    )
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
