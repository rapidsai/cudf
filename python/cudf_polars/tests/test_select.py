# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


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


@pytest.mark.parametrize("fmt", ["ndjson", "csv"])
def test_select_fast_count_unsupported_formats(tmp_path, fmt):
    df = pl.DataFrame({"a": [1, 2, 3]})
    file = tmp_path / f"test.{fmt}"
    if fmt == "csv":
        df.write_csv(file)
    elif fmt == "ndjson":
        df.write_ndjson(file)

    q = (
        pl.scan_csv(file).select(pl.len())
        if fmt == "csv"
        else pl.scan_ndjson(file).select(pl.len())
    )
    assert_ir_translation_raises(q, NotImplementedError)


def test_select_fast_count_parquet(tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3]})
    file = tmp_path / "data.parquet"
    df.write_parquet(file)

    q = pl.scan_parquet(file).select(pl.len())
    assert_gpu_result_equal(q)


@pytest.mark.parametrize(
    "zlice",
    [
        (1,),
        (1, 3),
        (-1,),
    ],
)
def test_select_fast_count_parquet_skip_rows(request, tmp_path, zlice):
    df = pl.DataFrame({"a": [1, 2, 3]})
    file = tmp_path / "data.parquet"
    df.write_parquet(file)

    q = pl.scan_parquet(file).slice(1, 5).select(pl.len())
    assert_gpu_result_equal(q)
