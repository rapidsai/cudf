# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.versions import POLARS_VERSION_LT_136


@pytest.fixture
def engine(streaming_engine_factory):
    return streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=3, fallback_mode="warn"),
    )


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


def test_over_in_filter_unsupported(request, streaming_engine_factory) -> None:
    engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=1, fallback_mode="warn"),
    )
    if not isinstance(engine, SPMDEngine):
        # On Dask/Ray the fallback warning fires on worker processes and is
        # invisible to ``pytest.warns``; the multi-rank fallback also
        # doesn't preserve row order.
        request.applymarker(
            pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/22405",
                strict=False,
            )
        )
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
