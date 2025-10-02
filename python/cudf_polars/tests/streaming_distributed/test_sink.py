# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl

from cudf_polars.testing.asserts import DEFAULT_SCHEDULER, assert_sink_result_equal
from cudf_polars.utils.config import ConfigOptions
from cudf_polars.utils.versions import POLARS_VERSION_LT_130


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
def test_sink_parquet_single_file(
    df, tmp_path, mkdir, data_page_size, row_group_size, max_rows_per_partition
):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": max_rows_per_partition,
            "scheduler": "synchronous",
            "sink_to_directory": False,
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


@pytest.mark.parametrize("mkdir", [True, False])
@pytest.mark.parametrize("data_page_size", [None, 256_000])
@pytest.mark.parametrize("row_group_size", [None, 1_000])
@pytest.mark.parametrize("max_rows_per_partition", [1_000, 1_000_000])
def test_sink_parquet_directory(
    df, tmp_path, mkdir, data_page_size, row_group_size, max_rows_per_partition
):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": max_rows_per_partition,
            "scheduler": DEFAULT_SCHEDULER,
            "sink_to_directory": True,
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
    assert check_path.is_dir()
    if expected_file_count > 1:
        assert len(list(check_path.iterdir())) == expected_file_count


def test_sink_parquet_distributed_raises():
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "scheduler": "distributed",
            "sink_to_directory": False,
        },
    )
    with pytest.raises(ValueError, match="distributed scheduler"):
        ConfigOptions.from_polars_engine(engine)


def test_sink_parquet_raises(df, tmp_path):
    if DEFAULT_SCHEDULER == "distributed":
        # We end up with an extra row per partition.
        pytest.skip("Distributed requires sink_to_directory=True")

    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 100_000,
            "scheduler": DEFAULT_SCHEDULER,
            "sink_to_directory": False,
        },
    )
    path = tmp_path / "test_sink_raises.parquet"
    df.sink_parquet(path, engine=engine)

    # Cannot overwrite an existing path with sink_to_directory=True
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 100_000,
            "scheduler": DEFAULT_SCHEDULER,
            "sink_to_directory": True,
        },
    )
    if POLARS_VERSION_LT_130:
        with pytest.raises(pl.exceptions.ComputeError, match="not supported"):
            df.sink_parquet(path, engine=engine)
    else:
        with pytest.raises(NotImplementedError, match="not supported"):
            df.sink_parquet(path, engine=engine)


@pytest.mark.parametrize("include_header", [True, False])
@pytest.mark.parametrize("null_value", [None, "NA"])
@pytest.mark.parametrize("separator", [",", "|"])
@pytest.mark.parametrize("max_rows_per_partition", [1_000, 1_000_000])
def test_sink_csv(
    df,
    tmp_path,
    include_header,
    null_value,
    separator,
    max_rows_per_partition,
):
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
        tmp_path / "out.csv",
        write_kwargs={
            "include_header": include_header,
            "null_value": null_value,
            "separator": separator,
        },
        read_kwargs={
            "has_header": include_header,
        },
        engine=engine,
    )


@pytest.mark.parametrize("max_rows_per_partition", [1_000, 1_000_000])
def test_sink_ndjson(df, tmp_path, max_rows_per_partition):
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
        tmp_path / "out.ndjson",
        engine=engine,
    )
