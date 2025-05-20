# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.dsl.ir import Join
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
    n_files = 3
    make_partitioned_source(df, tmp_path, "parquet", n_files=n_files)
    q = pl.scan_parquet(tmp_path)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "target_partition_size": 10_000,
            "scheduler": DEFAULT_SCHEDULER,
        },
    )

    q1 = q.select(pl.col("y"))
    qir1 = Translator(q1._ldf.visit(), engine).translate_ir()
    ir1, pi1 = lower_ir_graph(qir1, ConfigOptions(engine.config))
    table_stats_1 = pi1[ir1].table_stats
    element_size_y = table_stats_1.column_stats["y"].element_size
    assert table_stats_1.num_rows > 0
    assert element_size_y > 0

    q2 = q.filter(pl.col("z") < 3).group_by(pl.col("y")).mean().select(pl.col("x"))
    qir2 = Translator(q2._ldf.visit(), engine).translate_ir()
    ir2, pi2 = lower_ir_graph(qir2, ConfigOptions(engine.config))
    table_stats_2 = pi2[ir2].table_stats
    assert table_stats_2.num_rows > 0
    assert table_stats_2.column_stats["y"].element_size == element_size_y
    assert table_stats_2.column_stats["x"].element_size > 0


def test_table_statistics_join(tmp_path):
    # Left table
    tmp_dir_left = tmp_path / "temp_dir_left"
    tmp_dir_left.mkdir()
    left = pl.DataFrame(
        {
            "x": range(150),
            "y": range(0, 300, 2),
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 30,
        }
    )
    make_partitioned_source(left, tmp_dir_left, "parquet", n_files=1)
    dfl = pl.scan_parquet(tmp_dir_left)

    # Right table
    tmp_dir_right = tmp_path / "temp_dir_right"
    tmp_dir_right.mkdir()
    right = pl.DataFrame(
        {
            "xx": range(100),
            "y": list(range(50)) * 2,
            "zz": [1, 2, 3, 4, 5] * 20,
        }
    )
    make_partitioned_source(right, tmp_dir_right, "parquet", n_files=1)
    dfr = pl.scan_parquet(tmp_dir_right)

    # Make sure we get many partitions
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "target_partition_size": 500,
            "scheduler": DEFAULT_SCHEDULER,
            "shuffle_method": "tasks",
        },
    )

    # Check that we get the expected table stats
    # after a simple join.
    q = dfl.join(dfr, on="y", how="inner")
    qir = Translator(q._ldf.visit(), engine).translate_ir()
    ir, pi = lower_ir_graph(qir, ConfigOptions(engine.config))
    ir = ir if isinstance(ir, Join) else ir.children[0]
    table_stats_left = pi[ir.children[0]].table_stats
    table_stats_right = pi[ir.children[1]].table_stats
    table_stats = pi[ir].table_stats
    expected_num_rows = int(
        (table_stats_left.num_rows * table_stats_right.num_rows)
        / table_stats_left.column_stats["y"].unique_count
    )
    assert table_stats.num_rows == expected_num_rows
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)

    # Join on `q` again.
    # This provides test coverage for automatic repartitioning
    # of the smaller table (i.e. `q`).
    tmp_dir_right_2 = tmp_path / "temp_dir_right_2"
    tmp_dir_right_2.mkdir()
    right_2 = pl.DataFrame({"xx": range(100), "yy": range(0, 200, 2)})
    make_partitioned_source(right_2, tmp_dir_right_2, "parquet", n_files=1)
    dfr2 = pl.scan_parquet(tmp_dir_right_2)
    q2 = q.join(dfr2, on="xx", how="inner")
    assert_gpu_result_equal(q2, engine=engine, check_row_order=False)
