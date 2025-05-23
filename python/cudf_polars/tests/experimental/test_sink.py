# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import DEFAULT_SCHEDULER, assert_sink_result_equal
from cudf_polars.utils.versions import POLARS_VERSION_LT_128


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "x": range(30_000),
            "y": [1, 2, None] * 10_000,
            "z": ["ẅ", "a", "z", "123", "abcd"] * 6_000,
        }
    )


@pytest.mark.parametrize("mkdir", [True, False])
@pytest.mark.parametrize("data_page_size", [None, 256_000])
@pytest.mark.parametrize("row_group_size", [None, 1_000])
@pytest.mark.parametrize("max_rows_per_partition", [1_000, 1_000_000])
def test_sink_parquet(
    request, df, tmp_path, mkdir, data_page_size, row_group_size, max_rows_per_partition
):
    request.applymarker(
        pytest.mark.xfail(
            condition=POLARS_VERSION_LT_128,
            reason="not supported until polars 1.28",
        )
    )

    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": max_rows_per_partition,
            "scheduler": DEFAULT_SCHEDULER,
        },
    )

    assert_sink_result_equal(
        df,
        tmp_path / "test_sink.parquet",
        write_kwargs={
            "mkdir": mkdir,
            "data_page_size": data_page_size,
            "row_group_size": row_group_size,
        },
        engine=engine,
    )
