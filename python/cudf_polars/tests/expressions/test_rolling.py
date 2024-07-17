# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import polars as pl

from cudf_polars.testing.asserts import assert_ir_translation_raises


def test_rolling():
    dates = [
        "2020-01-01 13:45:48",
        "2020-01-01 16:42:13",
        "2020-01-01 16:45:09",
        "2020-01-02 18:12:48",
        "2020-01-03 19:45:32",
        "2020-01-08 23:16:43",
    ]
    df = (
        pl.DataFrame({"dt": dates, "a": [3, 7, 5, 9, 2, 1]})
        .with_columns(pl.col("dt").str.strptime(pl.Datetime))
        .lazy()
    )
    q = df.with_columns(
        sum_a=pl.sum("a").rolling(index_column="dt", period="2d"),
        min_a=pl.min("a").rolling(index_column="dt", period="2d"),
        max_a=pl.max("a").rolling(index_column="dt", period="2d"),
    )

    assert_ir_translation_raises(q, NotImplementedError)


def test_grouped_rolling():
    df = pl.LazyFrame({"a": [1, 2, 3, 4, 5, 6], "b": [1, 2, 1, 3, 1, 2]})

    q = df.select(pl.col("a").min().over("b"))

    assert_ir_translation_raises(q, NotImplementedError)
