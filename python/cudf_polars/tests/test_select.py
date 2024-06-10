# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_select():
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    ).lazy()

    query = ldf.select(
        pl.col("a") + pl.col("b"), (pl.col("a") * 2 + pl.col("b")).alias("d")
    )

    assert_gpu_result_equal(query)


def test_select_reduce():
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    ).lazy()

    query = ldf.select(
        (pl.col("a") + pl.col("b")).max(),
        (pl.col("a") * 2 + pl.col("b")).alias("d").mean(),
    )

    assert_gpu_result_equal(query)


def test_select_with_cse_no_agg():
    df = pl.LazyFrame({"a": [1, 2, 3]})
    expr = pl.col("a") + pl.col("a")

    query = df.select(expr, (expr * 2).alias("b"), ((expr * 2) + 10).alias("c"))

    assert_gpu_result_equal(query)


def test_select_with_cse_with_agg():
    df = pl.LazyFrame({"a": [1, 2, 3]})
    expr = pl.col("a") + pl.col("a")
    asum = pl.col("a").sum() + pl.col("a").sum()

    query = df.select(
        expr, (expr * 2).alias("b"), asum.alias("c"), (asum + 10).alias("d")
    )

    assert_gpu_result_equal(query)
