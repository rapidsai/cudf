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
def test_source_statistics(tmp_path, df, n_files, row_group_size):
    from cudf_polars.experimental.io import _extract_scan_stats

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
    )
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    column_stats = _extract_scan_stats(ir)

    # Source info is the same for all columns
    source = column_stats["x"].source
    assert source is column_stats["y"].source
    assert source is column_stats["z"].source
    assert source.row_count.value == df.height
    assert source.row_count.exact  # Sampled 3 files

    # Storage stats should be available
    assert source.storage_size("x").value > 0
    assert source.storage_size("y").value > 0

    # source._unique_stats should be empty
    assert set(source._unique_stats) == set()

    assert source.unique("x").count == df.height
    assert source.unique("x").fraction == 1.0

    # source._unique_stats should only contain 'x'
    assert set(source._unique_stats) == {"x"}

    # Mark 'z' as a key column, and query 'y' stats
    source.add_unique_stats_column("z")
    if n_files == 1 and row_group_size == 10_000:
        assert source.unique("y").count == 3
    else:
        assert source.unique("y").count is None
    assert source.unique("y").fraction < 1.0

    # source._unique_stats should contain all columns now
    assert set(source._unique_stats) == {"x", "y", "z"}
