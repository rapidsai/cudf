# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine
from cudf_polars.testing.asserts import assert_gpu_result_equal

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


def test_filter_pointwise(df, engine):
    query = df.filter(pl.col("a") > 3)
    assert_gpu_result_equal(query, engine=engine)


def test_filter_non_pointwise(df, engine):
    query = df.filter(pl.col("a") > pl.col("a").max())
    with pytest.warns(
        UserWarning, match="This filter is not supported for multiple partitions."
    ):
        assert_gpu_result_equal(query, engine=engine)
