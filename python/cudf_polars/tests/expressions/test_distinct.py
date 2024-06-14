# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture(
    params=["is_first_distinct", "is_last_distinct", "is_unique", "is_duplicated"]
)
def op(request):
    return request.param


@pytest.fixture
def df(with_nulls):
    values: list[int | None] = [1, 2, 3, 1, 1, 7, 3, 2, 7, 8, 1]
    if with_nulls:
        values[1] = None
        values[4] = None
    return pl.LazyFrame({"a": values})


def test_expr_distinct(df, op):
    expr = getattr(pl.col("a"), op)()
    query = df.select(expr)
    assert_gpu_result_equal(query)
