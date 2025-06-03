# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl

from cudf_polars.testing.asserts import DEFAULT_SCHEDULER, assert_sink_result_equal
from cudf_polars.utils.versions import POLARS_VERSION_LT_128, POLARS_VERSION_LT_130


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

    check_path = Path(tmp_path / "test_sink_gpu.parquet")
    expected_file_count = df.collect().height // max_rows_per_partition
    if expected_file_count > 1:
        assert check_path.is_dir()
        assert len(list(check_path.iterdir())) == expected_file_count
    else:
        assert not check_path.is_dir()


def test_sink_parquet_raises(request, df, tmp_path):
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
            "max_rows_per_partition": 100_000,
            "scheduler": DEFAULT_SCHEDULER,
        },
    )

    # We can write to a file once, but not twice
    path = tmp_path / "test_sink_raises.parquet"
    df.sink_parquet(path, engine=engine)
    if POLARS_VERSION_LT_130:
        with pytest.raises(pl.exceptions.ComputeError, match="not supported"):
            df.sink_parquet(path, engine=engine)
    else:
        with pytest.raises(NotImplementedError, match="not supported"):
            df.sink_parquet(path, engine=engine)
