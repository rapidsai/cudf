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
