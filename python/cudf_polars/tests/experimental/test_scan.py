# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.testing.asserts import DEFAULT_SCHEDULER, assert_gpu_result_equal
from cudf_polars.testing.io import make_partitioned_source
from cudf_polars.utils.config import ConfigOptions


@pytest.fixture(scope="module")
def df():
    return pl.DataFrame(
        {
            "x": range(3_000),
            "y": ["cat", "dog", "fish"] * 1_000,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 600,
        }
    )


@pytest.mark.parametrize(
    "fmt, scan_fn",
    [
        ("csv", pl.scan_csv),
        ("ndjson", pl.scan_ndjson),
        ("parquet", pl.scan_parquet),
    ],
)
def test_parallel_scan(tmp_path, df, fmt, scan_fn):
    make_partitioned_source(df, tmp_path, fmt, n_files=3)
    q = scan_fn(tmp_path)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"scheduler": DEFAULT_SCHEDULER},
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("blocksize", [1_000, 10_000, 1_000_000])
@pytest.mark.parametrize("n_files", [2, 3])
def test_target_partition_size(tmp_path, df, blocksize, n_files):
    make_partitioned_source(df, tmp_path, "parquet", n_files=n_files)
    q = pl.scan_parquet(tmp_path)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "target_partition_size": blocksize,
            "scheduler": DEFAULT_SCHEDULER,
        },
    )
    assert_gpu_result_equal(q, engine=engine)

    # Check partitioning
    qir = Translator(q._ldf.visit(), engine).translate_ir()
    ir, info = lower_ir_graph(qir, ConfigOptions.from_polars_engine(engine))
    count = info[ir].count
    if blocksize <= 12_000:
        assert count > n_files
    else:
        assert count < n_files


@pytest.mark.parametrize("mask", [None, pl.col("x") < 1_000])
def test_split_scan_predicate(tmp_path, df, mask):
    make_partitioned_source(df, tmp_path, "parquet", n_files=1)
    q = pl.scan_parquet(tmp_path)
    if mask is not None:
        q = q.filter(mask)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "target_partition_size": 1_000,
            "scheduler": DEFAULT_SCHEDULER,
        },
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("n_files", [1, 3])
@pytest.mark.parametrize("row_group_size", [None, 10_000])
@pytest.mark.parametrize("max_footer_samples", [3, 0])
@pytest.mark.parametrize("max_row_group_samples", [1, 0])
def test_source_statistics(
    tmp_path,
    df,
    n_files,
    row_group_size,
    max_footer_samples,
    max_row_group_samples,
):
    from cudf_polars.experimental.io import (
        _clear_source_info_cache,
        _extract_scan_stats,
    )

    _clear_source_info_cache()
    make_partitioned_source(
        df,
        tmp_path,
        "parquet",
        n_files=n_files,
        row_group_size=row_group_size,
    )
    q = pl.scan_parquet(tmp_path)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "target_partition_size": 10_000,
            "scheduler": DEFAULT_SCHEDULER,
        },
        parquet_options={
            "max_footer_samples": max_footer_samples,
            "max_row_group_samples": max_row_group_samples,
        },
    )
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    column_stats = _extract_scan_stats(ir, ConfigOptions.from_polars_engine(engine))

    # Source info is the same for all columns
    source_info = column_stats["x"].source_info
    assert source_info is column_stats["y"].source_info
    assert source_info is column_stats["z"].source_info
    if max_footer_samples:
        assert source_info.row_count.value == df.height
        assert source_info.row_count.exact
    else:
        assert source_info.row_count.value is None

    # Storage stats should be available
    if max_footer_samples:
        assert source_info.storage_size("x").value > 0
        assert source_info.storage_size("y").value > 0
    else:
        assert source_info.storage_size("x").value is None
        assert source_info.storage_size("y").value is None

    # Check that we can query a missing column name
    assert source_info.storage_size("foo").value is None
    assert source_info.unique_stats("foo").count.value is None
    assert source_info.unique_stats("foo").fraction.value is None

    # source._unique_stats should be empty
    assert set(source_info._unique_stats) == set()

    if max_footer_samples and max_row_group_samples:
        assert source_info.unique_stats("x").count.value == df.height
        assert source_info.unique_stats("x").fraction.value == 1.0
    else:
        assert source_info.unique_stats("x").count.value is None
        assert source_info.unique_stats("x").fraction.value is None

    # source_info._unique_stats should only contain 'x'
    if max_footer_samples and max_row_group_samples:
        assert set(source_info._unique_stats) == {"x"}
    else:
        assert set(source_info._unique_stats) == set()

    # Check add_unique_stats_column behavior
    if max_footer_samples and max_row_group_samples:
        # Can add a "bad"/missing key column
        source_info.add_unique_stats_column("foo")
        assert set(source_info._unique_stats) == {"x"}

        # Mark 'z' as a key column, and query 'y' stats
        source_info.add_unique_stats_column("z")
        if n_files == 1 and row_group_size == 10_000:
            assert source_info.unique_stats("y").count.value == 3
        else:
            assert source_info.unique_stats("y").count.value is None
        assert source_info.unique_stats("y").fraction.value < 1.0

        # source_info._unique_stats should contain all columns now
        assert set(source_info._unique_stats) == {"x", "y", "z"}


def test_source_statistics_csv(tmp_path, df):
    from cudf_polars.experimental.io import _extract_scan_stats

    make_partitioned_source(df, tmp_path, "csv", n_files=3)
    q = pl.scan_csv(tmp_path)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "target_partition_size": 10_000,
            "scheduler": DEFAULT_SCHEDULER,
        },
    )
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    column_stats = _extract_scan_stats(ir, ConfigOptions.from_polars_engine(engine))

    # Source info should be empty for CSV
    source_info = column_stats["x"].source_info
    assert source_info.row_count.value is None
    assert source_info.unique_stats("x").count.value is None
    assert source_info.unique_stats("x").fraction.value is None
