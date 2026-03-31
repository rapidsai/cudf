# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)
from cudf_polars.utils.versions import (
    POLARS_VERSION_LT_132,
    POLARS_VERSION_LT_134,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from cudf_polars.experimental.rapidsmpf.frontend.core import StreamingEngine


@pytest.fixture
def engine(
    request: pytest.FixtureRequest,
) -> Generator[StreamingEngine, None, None]:
    params: dict[str, Any] = getattr(request, "param", {})
    executor_options = {
        "max_rows_per_partition": 3,
        "fallback_mode": "warn",
        "dynamic_planning": {},
        **params.get("executor_options", {}),
    }
    with SPMDEngine(executor_options=executor_options) as engine:
        yield engine


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


@pytest.mark.parametrize(
    "fallback_mode,engine",
    [
        ("silent", {"executor_options": {"fallback_mode": "silent"}}),
        ("raise", {"executor_options": {"fallback_mode": "raise"}}),
        ("warn", {"executor_options": {"fallback_mode": "warn"}}),
        ("foo", {"executor_options": {"fallback_mode": "foo"}}),
    ],
    indirect=["engine"],
)
def test_select_reduce_fallback(df, fallback_mode, engine):
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
            NotImplementedError,
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


def test_select_fill_null_with_strategy(df, engine):
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


@pytest.mark.parametrize(
    "aggs",
    [
        (pl.col("a").drop_nulls().n_unique(),),
        (pl.col("a").drop_nulls().sum(),),
        (pl.col("a").drop_nulls().min(),),
        (pl.col("a").drop_nulls().max(),),
        (pl.col("a").drop_nulls().mean(),),
    ],
)
def test_select_drop_nulls_aggs(engine, aggs):
    df = pl.LazyFrame({"a": [1, 2, None, 2, 3, None, 1, 4]})
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


def test_select_with_empty_partitions(engine):
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


def test_select_mean_with_decimals(engine):
    df = pl.LazyFrame({"d": [Decimal("1.23")] * 4})
    q = df.select(pl.mean("d"))
    assert_gpu_result_equal(q, engine=engine, check_dtypes=not POLARS_VERSION_LT_134)


def test_select_with_len(engine):
    # https://github.com/pola-rs/polars/issues/25592
    df1 = pl.LazyFrame({"c0": [1] * 4})
    df2 = pl.LazyFrame({"c0": [2] * 4})
    q = pl.concat([df1.join(df2, how="cross"), df1.with_columns(pl.lit(None))]).select(
        pl.len()
    )
    with pytest.warns(
        UserWarning, match="Cross join not support for multiple partitions"
    ):
        assert_gpu_result_equal(q, engine=engine)


def test_select_with_mixed_fusable_non_fusable_exprs(df, engine):
    q = df.select(
        foo=pl.col("a").n_unique(),
        bar=pl.col("b").sum(),
        baz=pl.col("c").sum(),
    )
    assert_gpu_result_equal(q, engine=engine)
