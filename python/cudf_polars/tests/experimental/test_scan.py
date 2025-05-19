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
    ir, info = lower_ir_graph(qir, ConfigOptions(engine.config))
    count = info[ir].count
    if blocksize <= 12_000:
        assert count > n_files
    else:
        assert count < n_files


def test_table_statistics(tmp_path, df):
    make_partitioned_source(df, tmp_path, "parquet", n_files=3)
    q = pl.scan_parquet(tmp_path)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "target_partition_size": 10_000,
            "scheduler": DEFAULT_SCHEDULER,
        },
    )

    q = q.filter(pl.col("z") < 3).group_by(pl.col("y")).mean().select(pl.col("x"))

    # Check that we are tracking TableStats,
    # and that the basic stats make sense
    qir = Translator(q._ldf.visit(), engine).translate_ir()
    ir, pi = lower_ir_graph(qir, ConfigOptions(engine.config))
    table_stats = pi[ir].table_stats
    result = q.collect()
    assert table_stats.num_rows >= len(result)
    assert table_stats.column_stats["y"].unique_count == len(result)
    assert table_stats.column_stats["y"].element_size > 0
