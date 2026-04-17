# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.versions import POLARS_VERSION_LT_136

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


def test_rolling_datetime(request, engine):
    if not POLARS_VERSION_LT_136:
        request.applymarker(
            pytest.mark.xfail(reason="See https://github.com/pola-rs/polars/pull/25117")
        )
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
        .with_columns(pl.col("dt").str.strptime(pl.Datetime("ns")))
        .lazy()
    )
    q = df.with_columns(pl.sum("a").rolling(index_column="dt", period="2d"))
    # HStack may redirect to Select before fallback; message differs by Polars IR / version.
    with pytest.warns(
        UserWarning,
        match=r"This (HStack|selection) is not supported for multiple partitions\.",
    ):
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over("g"),
        pl.len().over("g"),
        pl.col("x").sum().over("g", "g2"),
        pl.col("x").sum().over("g_null"),
        pl.col("x").sum().over("g", order_by="s"),
        pl.col("x").rank(method="dense", descending=True).over("g"),
        pl.col("x").rank(method="min").over("g", "g2"),
        pl.col("x").cum_sum().over("g", order_by="s"),
        pl.when((pl.col("x") % 2) == 0)
        .then(None)
        .otherwise(pl.col("x"))
        .fill_null(strategy="forward")
        .over("g", order_by="s"),
    ],
    ids=[
        "single_key_sum",
        "len_over",
        "multi_key",
        "null_keys",
        "order_by",
        "rank_dense",
        "rank_min_multi_key",
        "cum_sum_order_by",
        "fill_null_forward",
    ],
)
def test_over_select(engine, expr):
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "x": [1, 2, 3, 4, 5, 6],
            "g2": ["a", "b", "a", "b", "a", "b"],
            "g_null": [1, None, 1, None, 2, 1],
            "s": [6, 5, 4, 3, 2, 1],
        }
    )
    assert_gpu_result_equal(df.select(expr), engine=engine, check_row_order=True)


def test_over_with_columns(engine):
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )
    assert_gpu_result_equal(
        df.with_columns(pl.col("x").sum().over("g")),
        engine=engine,
        check_row_order=True,
    )


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over(pl.col("g") % 2),
        pl.col("x").sum().over("g", pl.col("x") % 2),
    ],
    ids=["noncol_key", "mixed_col_and_expr_key"],
)
@pytest.mark.parametrize(
    "engine",
    [{"executor_options": {"max_rows_per_partition": 2}}],
    indirect=True,
)
def test_over_noncol_key_fallback(engine, expr) -> None:
    # Non-Col and mixed Col/expr partition-by keys are not yet supported for
    # multi-partition streaming and should fall back to single-partition.
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )
    with pytest.warns(UserWarning, match=r"not supported for multiple partitions"):
        assert_gpu_result_equal(df.select(expr), engine=engine)


@pytest.mark.parametrize(
    "engine",
    [{"executor_options": {"max_rows_per_partition": 2}}],
    indirect=True,
)
def test_over_mixed_keys_fallback(engine) -> None:
    # Multiple over expressions with different partition by keys
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "g2": ["a", "b", "a", "b", "a", "b"],
            "x": [1, 2, 3, 4, 5, 6],
        }
    )
    q = df.select(
        pl.col("x").sum().over("g").alias("s_g"),
        pl.col("x").sum().over("g2").alias("s_g2"),
    )
    with pytest.warns(UserWarning, match=r"not supported for multiple partitions"):
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over("g"),
        pl.len().over("g"),
        pl.col("x").rank(method="dense").over("g"),
        pl.col("x").cum_sum().over("g", order_by="s"),
    ],
    ids=["scalar_sum", "scalar_len", "nonscalar_rank", "nonscalar_cum_sum"],
)
@pytest.mark.parametrize(
    "engine",
    [{"executor_options": {"max_rows_per_partition": 1}}],
    indirect=True,
)
def test_over_many_partitions(engine, expr) -> None:
    # max_rows_per_partition=1 forces one chunk per row, exercising the AllGather
    # (scalar broadcast) and sort-and-split (non-scalar) paths across many partitions.
    df = pl.LazyFrame(
        {
            "g": [1, 1, 2, 2, 2, 1],
            "x": [1, 2, 3, 4, 5, 6],
            "s": [6, 5, 4, 3, 2, 1],
        }
    )
    assert_gpu_result_equal(df.select(expr), engine=engine, check_row_order=True)


@pytest.mark.parametrize(
    "expr",
    [
        pl.col("x").sum().over("g"),
        pl.col("x").rank(method="dense").over("g"),
    ],
    ids=["scalar_sum", "nonscalar_rank"],
)
@pytest.mark.parametrize(
    "engine",
    [{"executor_options": {"max_rows_per_partition": 2}}],
    indirect=True,
)
def test_over_empty_input(engine, expr) -> None:
    df = pl.LazyFrame(
        {
            "g": pl.Series([], dtype=pl.Int64),
            "x": pl.Series([], dtype=pl.Int64),
        }
    )
    assert_gpu_result_equal(df.select(expr), engine=engine, check_row_order=True)


@pytest.mark.parametrize(
    "engine",
    [{"executor_options": {"max_rows_per_partition": 1}}],
    indirect=True,
)
def test_over_in_filter_unsupported(engine) -> None:
    q = pl.concat(
        [
            pl.LazyFrame({"k": ["x", "y"], "v": [3, 2]}),
            pl.LazyFrame({"k": ["x", "y"], "v": [5, 7]}),
        ]
    ).filter(pl.len().over("k") == 2)

    with pytest.warns(
        UserWarning,
        match=r"over\(...\) inside filter is not supported for multiple partitions.*",
    ):
        assert_gpu_result_equal(q, engine=engine)
