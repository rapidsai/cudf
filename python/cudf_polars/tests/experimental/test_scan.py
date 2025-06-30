# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.experimental.statistics import collect_source_stats
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


@pytest.mark.parametrize("parquet_metadata_samples", [1, 3])
@pytest.mark.parametrize("parquet_rowgroup_samples", [1, 2])
def test_column_source_statistics(
    tmp_path,
    df,
    parquet_metadata_samples,
    parquet_rowgroup_samples,
):
    n_files = 3
    make_partitioned_source(df, tmp_path, "parquet", n_files=n_files)
    q = pl.scan_parquet(tmp_path)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "target_partition_size": 10_000,
            "scheduler": DEFAULT_SCHEDULER,
            "parquet_metadata_samples": parquet_metadata_samples,
            "parquet_rowgroup_samples": parquet_rowgroup_samples,
        },
    )
    q1 = q.select(pl.col("x"), pl.col("y"))
    qir1 = Translator(q1._ldf.visit(), engine).translate_ir()
    stats = collect_source_stats(qir1, ConfigOptions.from_polars_engine(engine))
    source_stats_y = stats.column_stats[qir1]["y"].source_stats
    y_unique_fraction = source_stats_y.unique_fraction
    y_cardinality = source_stats_y.cardinality
    assert y_unique_fraction < 1.0
    assert y_unique_fraction > 0.0
    if parquet_metadata_samples >= n_files:
        # We should have "exact" cardinality statistics
        assert y_cardinality == df.height
        assert "cardinality" in source_stats_y.exact
    else:
        # We should have "estimated" cardinality statistics
        assert y_cardinality > 0
        assert "cardinality" not in source_stats_y.exact
    assert_gpu_result_equal(q1.sort(pl.col("x")).slice(0, 2), engine=engine)

    # Source statistics of "y" should match after GroupBy/Select/HStack/etc
    q2 = (
        pl.concat(
            [
                q.select(pl.col("x")),
                q.select(pl.col("y")),
            ],
            how="horizontal",
        )
        .group_by(pl.col("y"))
        .sum()
        .select(pl.col("x").max(), pl.col("y"))
        .with_columns((pl.col("x") * pl.col("x")).alias("x2"))
    )
    qir2 = Translator(q2._ldf.visit(), engine).translate_ir()
    stats = collect_source_stats(qir2, ConfigOptions.from_polars_engine(engine))
    source_stats_y = stats.column_stats[qir2]["y"].source_stats
    assert source_stats_y.unique_fraction == y_unique_fraction
    assert y_cardinality == source_stats_y.cardinality
    assert_gpu_result_equal(q2.sort(pl.col("y")).slice(0, 2), engine=engine)
