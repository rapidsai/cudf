# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from decimal import Decimal

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import (
    POLARS_VERSION_LT_130,
    POLARS_VERSION_LT_132,
    POLARS_VERSION_LT_134,
)


@pytest.fixture(scope="module")
def engine():
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
            "c": [2, 4, 6, 8, 10, 12, 14],
        }
    )


def test_select(df, engine):
    query = df.select(
        pl.col("a") + pl.col("b"), (pl.col("a") * 2 + pl.col("b")).alias("d")
    )
    assert_gpu_result_equal(query, engine=engine)


@pytest.mark.parametrize("fallback_mode", ["silent", "raise", "warn", "foo"])
def test_select_reduce_fallback(df, fallback_mode):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "fallback_mode": fallback_mode,
            "max_rows_per_partition": 3,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )
    match = "This selection is not supported for multiple partitions."

    query = df.select(
        (pl.col("a") + pl.col("b")).max(),
        # NOTE: We don't support `median` yet
        (pl.col("a") * 2 + pl.col("b")).alias("d").median(),
    )

    if fallback_mode == "silent":
        ctx = contextlib.nullcontext()
    elif fallback_mode == "raise":
        ctx = pytest.raises(
            pl.exceptions.ComputeError
            if POLARS_VERSION_LT_130
            else NotImplementedError,
            match=match,
        )
    elif fallback_mode == "foo":
        ctx = pytest.raises(
            pl.exceptions.ComputeError,
            match="'foo' is not a valid StreamingFallbackMode",
        )
    else:
        ctx = pytest.warns(UserWarning, match=match)
    with ctx:
        assert_gpu_result_equal(query, engine=engine)


def test_select_fill_null_with_strategy(df):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "fallback_mode": "warn",
            "max_rows_per_partition": 3,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )
    q = df.select(pl.col("a").forward_fill())

    if POLARS_VERSION_LT_132:
        assert_ir_translation_raises(q, NotImplementedError)
    else:
        with pytest.warns(
            UserWarning,
            match="fill_null with strategy other than 'zero' or 'one' is not supported for multiple partitions",
        ):
            assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "aggs",
    [
        (pl.col("a").sum(),),
        (pl.col("a").min() + pl.col("a"),),
        (
            (pl.col("a") + pl.col("b")).sum(),
            (pl.col("a") * 2 + pl.col("b")).alias("d").min(),
        ),
        (pl.col("a").min() + pl.col("b").max(),),
        (pl.col("a") - (pl.col("b") + pl.col("c").max()).sum(),),
        (pl.col("b").len(),),
        (pl.col("a") - (pl.col("b") + pl.col("c").max()).mean(),),
        (
            pl.col("b").sum(),
            (pl.col("c").sum() + 1),
        ),
        (
            pl.col("b").n_unique(),
            (pl.col("c").n_unique() + 1),
        ),
        (pl.col("a").min(), pl.col("b"), pl.col("c").max()),
    ],
)
def test_select_aggs(df, engine, aggs):
    # Test supported aggs (e.g. "min", "max", "mean", "n_unique")
    query = df.select(*aggs)
    assert_gpu_result_equal(query, engine=engine)


def test_select_with_cse_no_agg(df, engine):
    expr = pl.col("a") + pl.col("a")
    query = df.select(expr, (expr * 2).alias("b"), ((expr * 2) + 10).alias("c"))
    assert_gpu_result_equal(query, engine=engine)


def test_select_parquet_fast_count(tmp_path, df, engine):
    file = tmp_path / "data.parquet"
    df.collect().write_parquet(file)
    q = pl.scan_parquet(file).select(pl.len())
    assert_gpu_result_equal(q, engine=engine)


def test_select_literal(engine):
    # See: https://github.com/rapidsai/cudf/issues/19147
    ldf = pl.LazyFrame({"a": list(range(10))})
    q = ldf.select(pl.lit(2).pow(pl.lit(-3, dtype=pl.Float32)))
    assert_gpu_result_equal(q, engine=engine)


def test_select_with_empty_partitions(df, engine):
    df = pl.concat(
        [
            pl.LazyFrame({"b": pl.Series([], dtype=pl.Decimal(15, 2))}),
            pl.LazyFrame({"b": pl.Series([], dtype=pl.Decimal(15, 2))}),
        ]
    )
    q = df.select(pl.col("b").sum() / Decimal("7.00"))
    # Polars pre their decimal overhaul: https://github.com/pola-rs/polars/issues/19784
    # returned a different precision and scale, so we skip dtype check
    assert_gpu_result_equal(q, engine=engine, check_dtypes=not POLARS_VERSION_LT_134)
