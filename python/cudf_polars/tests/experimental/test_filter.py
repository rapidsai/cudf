# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture
def engine(streaming_engine_factory):
    # ``fallback_mode="warn"`` overrides the small-blocksize baseline (which
    # sets SILENT) so ``test_filter_non_pointwise`` can assert on the warning.
    return streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=3, fallback_mode="warn"),
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


def test_filter_pointwise(df, engine):
    query = df.filter(pl.col("a") > 3)
    assert_gpu_result_equal(query, engine=engine)


def test_filter_non_pointwise(df, engine):
    query = df.filter(pl.col("a") > pl.col("a").max())
    with pytest.warns(
        UserWarning, match="This filter is not supported for multiple partitions."
    ):
        assert_gpu_result_equal(query, engine=engine)
