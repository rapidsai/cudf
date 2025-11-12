# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
    assert_gpu_result_equal,
)
from cudf_polars.utils.config import StreamingFallbackMode


def test_rolling_datetime():
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
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "fallback_mode": StreamingFallbackMode.WARN,
        },
    )
    q = df.with_columns(pl.sum("a").rolling(index_column="dt", period="2d"))
    with pytest.warns(
        UserWarning, match="This HStack not supported for multiple partitions."
    ):
        assert_gpu_result_equal(q, engine=engine)


def test_over_in_filter_unsupported() -> None:
    q = pl.concat(
        [
            pl.LazyFrame({"k": ["x", "y"], "v": [3, 2]}),
            pl.LazyFrame({"k": ["x", "y"], "v": [5, 7]}),
        ]
    ).filter(pl.len().over("k") == 2)

    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 1,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "fallback_mode": StreamingFallbackMode.WARN,
        },
    )
    with pytest.warns(
        UserWarning,
        match=r"over\(...\) inside filter is not supported for multiple partitions.*",
    ):
        assert_gpu_result_equal(q, engine=engine)
