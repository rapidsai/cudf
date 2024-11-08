# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from contextlib import nullcontext

import pytest

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_112, POLARS_VERSION_LT_113


@pytest.fixture(params=[False, True], ids=["nulls_not_equal", "nulls_equal"])
def join_nulls(request):
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
    "join_expr",
    [
        pl.col("a"),
        pl.col("a") * 2,
        [pl.col("a"), pl.col("c") + 1],
        ["c", "a"],
    ],
)
def test_non_coalesce_join(left, right, how, join_nulls, join_expr):
    query = left.join(
        right, on=join_expr, how=how, join_nulls=join_nulls, coalesce=False
    )
    assert_gpu_result_equal(query, check_row_order=how == "left")


@pytest.mark.parametrize(
    "join_expr",
    [
        pl.col("a"),
        ["c", "a"],
    ],
)
def test_coalesce_join(left, right, how, join_nulls, join_expr):
    query = left.join(
        right, on=join_expr, how=how, join_nulls=join_nulls, coalesce=True
    )
    assert_gpu_result_equal(query, check_row_order=how == "left")


def test_left_join_with_slice(left, right, join_nulls, zlice):
    q = left.join(right, on="a", how="left", join_nulls=join_nulls, coalesce=True)
    ctx = nullcontext()
    if zlice is not None:
        q_expect = q.collect().slice(*zlice)
        q = q.slice(*zlice)
        if POLARS_VERSION_LT_112 and (zlice == (1, 5) or zlice == (0, 2)):
            # https://github.com/pola-rs/polars/issues/19403
            # https://github.com/pola-rs/polars/issues/19405
            ctx = pytest.raises(AssertionError)
            assert_frame_equal(
                q_expect, q.collect(engine=pl.GPUEngine(raise_on_fail=True))
            )

    with ctx:
        assert_gpu_result_equal(q)


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
def test_join_literal_key_unsupported(left, right, left_on, right_on):
    q = left.join(right, left_on=left_on, right_on=right_on, how="inner")

    assert_ir_translation_raises(q, NotImplementedError)


@pytest.mark.parametrize(
    "conditions",
    [
        [pl.col("a") < pl.col("a_right")],
        [pl.col("a_right") <= pl.col("a") * 2],
        [pl.col("b") * 2 > pl.col("a_right"), pl.col("a") == pl.col("c_right")],
        [pl.col("b") * 2 <= pl.col("a_right"), pl.col("a") < pl.col("c_right")],
        pytest.param(
            [pl.col("b") <= pl.col("a_right") * 7, pl.col("a") < pl.col("d") * 2],
            marks=pytest.mark.xfail(
                POLARS_VERSION_LT_113,
                reason="https://github.com/pola-rs/polars/issues/19597",
            ),
        ),
    ],
)
def test_join_where(left, right, conditions, zlice):
    q = left.join_where(right, *conditions)

    assert_gpu_result_equal(q, check_row_order=False)

    if zlice is not None:
        q_len = q.slice(*zlice).select(pl.len())
        # Can't compare result, since row order is not guaranteed and
        # therefore we only check the length

        assert_gpu_result_equal(q_len)
