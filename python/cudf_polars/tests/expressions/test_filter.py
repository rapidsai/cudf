# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "expr",
    [
        pytest.param(
            pl.lit(value=False),
            marks=pytest.mark.xfail(reason="Expression filter does not handle scalars"),
        ),
        pl.col("c"),
        pl.col("b") > 2,
    ],
)
@pytest.mark.parametrize("predicate_pushdown", [False, True])
def test_filter_expression(expr, predicate_pushdown):
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [0, 3, 1, 5, 6, 1, 0],
            "c": [None, True, False, False, True, True, False],
        }
    )

    query = ldf.select(pl.col("a").filter(expr))
    assert_gpu_result_equal(
        query, collect_kwargs={"predicate_pushdown": predicate_pushdown}
    )
