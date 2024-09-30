# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


def test_union():
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    ).lazy()
    ldf2 = ldf.select((pl.col("a") + pl.col("b")).alias("c"), pl.col("a"))
    query = pl.concat([ldf, ldf2], how="diagonal")
    assert_gpu_result_equal(query)


def test_union_schema_mismatch_raises():
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    ).lazy()
    ldf2 = ldf.select(pl.col("a").cast(pl.Float32))
    query = pl.concat([ldf, ldf2], how="diagonal")

    assert_ir_translation_raises(query, NotImplementedError)


def test_concat_vertical():
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    )
    ldf2 = ldf.select(pl.col("a"), pl.col("b") * 2 + pl.col("a"))
    q = pl.concat([ldf, ldf2], how="vertical")

    assert_gpu_result_equal(q)


def test_concat_diagonal_empty():
    df1 = pl.LazyFrame()
    df2 = pl.LazyFrame({"a": [1, 2]})

    q = pl.concat([df1, df2], how="diagonal_relaxed")

    assert_gpu_result_equal(q, collect_kwargs={"no_optimization": True})
