# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl

from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
from cudf_polars.testing.asserts import assert_sink_result_equal
from cudf_polars.utils.config import ConfigOptions


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
    df,
    streaming_engine_factory,
    tmp_path,
    mkdir,
    data_page_size,
    row_group_size,
    max_rows_per_partition,
):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=max_rows_per_partition),
    )
    assert_sink_result_equal(
        df,
        tmp_path / "test_sink.parquet",
        write_kwargs={
            "mkdir": mkdir,
            "data_page_size": data_page_size,
            "row_group_size": row_group_size,
        },
        engine=streaming_engine,
    )


@pytest.mark.parametrize("mkdir", [True, False])
@pytest.mark.parametrize("data_page_size", [None, 256_000])
@pytest.mark.parametrize("row_group_size", [None, 1_000])
@pytest.mark.parametrize("max_rows_per_partition", [1_000, 1_000_000])
def test_sink_parquet_directory(
    df,
    streaming_engine_factory,
    tmp_path,
    mkdir,
    data_page_size,
    row_group_size,
    max_rows_per_partition,
):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=max_rows_per_partition,
            sink_to_directory=True,
        ),
    )
    assert_sink_result_equal(
        df,
        tmp_path / "test_sink.parquet",
        write_kwargs={
            "mkdir": mkdir,
            "data_page_size": data_page_size,
            "row_group_size": row_group_size,
        },
        engine=streaming_engine,
    )

    check_path = Path(tmp_path / "test_sink_gpu.parquet")
    expected_file_count = (
        df.collect(engine=streaming_engine).height // max_rows_per_partition
    )
    assert check_path.is_dir()
    if expected_file_count > 1:
        assert len(list(check_path.iterdir())) == expected_file_count


def test_sink_parquet_raises_spmd(spmd_engine):
    from cudf_polars.experimental.rapidsmpf.frontend.spmd import SPMDEngine

    with (
        pytest.raises(
            ValueError, match="The spmd cluster requires sink_to_directory=True"
        ),
        SPMDEngine(
            comm=spmd_engine.comm, executor_options={"sink_to_directory": False}
        ) as engine,
    ):
        ConfigOptions.from_polars_engine(engine)


def test_sink_parquet_raises(df, tmp_path):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 100_000,
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
            "sink_to_directory": True,
        },
    )
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
        },
    )

    assert_sink_result_equal(
        df,
        tmp_path / "out.ndjson",
        engine=engine,
    )
