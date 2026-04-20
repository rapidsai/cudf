# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for distributed LazyFrame.rolling().agg() via the streaming engine."""

from __future__ import annotations

from datetime import datetime
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


@pytest.fixture
def engine_rolling(
    request: pytest.FixtureRequest,
) -> Generator[StreamingEngine, None, None]:
    """Like ``engine`` but omits ``fallback_mode`` so unsupported nodes fail instead of warning."""
    params: dict[str, Any] = getattr(request, "param", {})
    executor_options = {
        "max_rows_per_partition": 3,
        "dynamic_planning": {},
        **params.get("executor_options", {}),
    }
    with SPMDEngine(executor_options=executor_options) as engine:
        yield engine


def test_rolling_datetime(request, engine):
    """Expression-level rolling via with_columns falls back with a warning."""
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


@pytest.mark.parametrize("closed", ["left", "right", "both", "none"])
def test_ungrouped_int_rolling_closed_variants(engine_rolling, closed) -> None:
    """All four closed-window variants on an integer index crossing partition boundaries."""
    df = pl.LazyFrame(
        {
            "idx": [1, 2, 3, 4, 5, 6, 7, 8],
            "val": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )
    q = df.rolling("idx", period="3i", closed=closed).agg(
        s=pl.col("val").sum(),
        m=pl.col("val").min(),
        x=pl.col("val").max(),
        n=pl.len(),
    )
    assert_gpu_result_equal(q, engine=engine_rolling)


def test_ungrouped_int_rolling_multiple_aggs(engine_rolling) -> None:
    df = pl.LazyFrame(
        {
            "idx": [1, 2, 4, 5, 7, 8, 10, 11],
            "a": [1, 2, 3, 4, 5, 6, 7, 8],
            "b": [8, 7, 6, 5, 4, 3, 2, 1],
        }
    )
    q = df.rolling("idx", period="4i", closed="right").agg(
        a_sum=pl.col("a").sum(),
        b_min=pl.col("b").min(),
        n=pl.len(),
    )
    assert_gpu_result_equal(q, engine=engine_rolling)


def test_ungrouped_int_rolling_right_halo_spans_multiple_chunks(engine_rolling) -> None:
    """Positive lookahead can pull the right halo from more than one next partition."""
    df = pl.LazyFrame(
        {
            "idx": list(range(1, 13)),
            "val": list(range(1, 13)),
        }
    )
    q = df.rolling("idx", period="6i", offset="-1i", closed="right").agg(
        s=pl.col("val").sum(),
    )
    assert_gpu_result_equal(q, engine=engine_rolling)


def test_ungrouped_int_rolling_lookback_spans_multiple_prior_chunks(
    engine_rolling,
) -> None:
    """Default offset gives a wide lookback so the left halo spans several prior partitions."""
    df = pl.LazyFrame(
        {
            "idx": list(range(1, 16)),
            "val": list(range(1, 16)),
        }
    )
    q = df.rolling("idx", period="8i", closed="right").agg(s=pl.col("val").sum())
    assert_gpu_result_equal(q, engine=engine_rolling)


def test_ungrouped_int_rolling_halo_spans_multiple_chunks_lookahead_and_lookback(
    engine_rolling,
) -> None:
    """``offset=-5i`` with ``period=10i`` needs wide halos on both sides across several partitions."""
    df = pl.LazyFrame(
        {
            "idx": list(range(1, 16)),
            "val": list(range(1, 16)),
        }
    )
    q = df.rolling("idx", period="10i", offset="-5i", closed="right").agg(
        s=pl.col("val").sum(),
    )
    assert_gpu_result_equal(q, engine=engine_rolling)


@pytest.mark.parametrize("period", ["2d", "4d"])
def test_ungrouped_datetime_rolling(engine_rolling, period) -> None:
    dates = [
        datetime(2020, 1, 1),
        datetime(2020, 1, 2),
        datetime(2020, 1, 3),
        datetime(2020, 1, 5),
        datetime(2020, 1, 7),
        datetime(2020, 1, 8),
        datetime(2020, 1, 10),
    ]
    df = pl.LazyFrame(
        {
            "dt": pl.Series(dates, dtype=pl.Datetime("us")),
            "val": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    q = df.rolling("dt", period=period, closed="right").agg(
        s=pl.col("val").sum(),
        n=pl.len(),
    )
    assert_gpu_result_equal(q, engine=engine_rolling)


def test_grouped_rolling_integer_index(engine_rolling) -> None:
    """Grouped rolling where group members span partition boundaries."""
    df = pl.LazyFrame(
        {
            "key": ["A", "A", "B", "B", "A", "B", "A", "B"],
            "idx": [1, 2, 1, 2, 3, 3, 4, 4],
            "val": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )
    q = df.rolling("idx", period="3i", group_by="key", closed="right").agg(
        s=pl.col("val").sum(),
        n=pl.len(),
    )
    assert_gpu_result_equal(q, engine=engine_rolling, check_row_order=False)


def test_grouped_rolling_multiple_keys(engine_rolling) -> None:
    df = pl.LazyFrame(
        {
            "k1": ["X", "X", "Y", "Y", "X", "Y", "X", "Y"],
            "k2": ["a", "a", "b", "b", "a", "b", "a", "b"],
            "idx": [1, 2, 1, 2, 3, 3, 4, 4],
            "val": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    q = df.rolling("idx", period="2i", group_by=["k1", "k2"], closed="both").agg(
        s=pl.col("val").sum(),
    )
    assert_gpu_result_equal(q, engine=engine_rolling, check_row_order=False)


def test_rolling_with_slice(engine_rolling) -> None:
    df = pl.LazyFrame(
        {
            "idx": [1, 2, 3, 4, 5, 6, 7, 8],
            "val": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )
    q = (
        df.rolling("idx", period="3i", closed="right")
        .agg(
            s=pl.col("val").sum(),
        )
        .slice(2, 4)
    )
    assert_gpu_result_equal(q, engine=engine_rolling)


def test_rolling_no_cross_partition_halo(engine_rolling) -> None:
    """Narrow window and a wide index gap so halos never reach the next partition."""
    df = pl.LazyFrame(
        {
            "idx": [1, 2, 3, 10, 11, 12],
            "val": [1, 2, 3, 4, 5, 6],
        }
    )
    q = df.rolling("idx", period="1i", closed="right").agg(s=pl.col("val").sum())
    assert_gpu_result_equal(q, engine=engine_rolling)
