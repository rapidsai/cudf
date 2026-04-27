# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.experimental.statistics import collect_statistics
from cudf_polars.testing.asserts import assert_gpu_result_equal
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
def test_parallel_scan(tmp_path, df, fmt, scan_fn, streaming_engine):
    make_partitioned_source(df, tmp_path, fmt, n_files=3)
    q = scan_fn(tmp_path)
    assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize(
    "streaming_engine",
    [
        {
            "executor_options": {"target_partition_size": 1_000},
            "engine_options": {"parquet_options": {"use_rapidsmpf_native": True}},
        }
    ],
    indirect=True,
)
def test_scan_parquet_use_rapidsmpf_native(tmp_path, df, streaming_engine):
    make_partitioned_source(df, tmp_path, "parquet", n_files=1)
    assert_gpu_result_equal(pl.scan_parquet(tmp_path), engine=streaming_engine)


# ---------------------------------------------------------------------------
# Tests migrated from tests/experimental/test_scan.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "streaming_engine",
    [{"executor_options": {"target_partition_size": 1_000}}],
    indirect=True,
)
def test_split_scan_aligns_to_row_group_boundaries(tmp_path, df, streaming_engine):
    make_partitioned_source(df, tmp_path, "parquet", n_files=1, row_group_size=10)
    q = pl.scan_parquet(tmp_path)
    assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize("mask", [None, pl.col("x") < 1_000])
@pytest.mark.parametrize(
    "streaming_engine",
    [{"executor_options": {"target_partition_size": 1_000}}],
    indirect=True,
)
def test_split_scan_predicate(tmp_path, df, mask, streaming_engine):
    make_partitioned_source(df, tmp_path, "parquet", n_files=1)
    q = pl.scan_parquet(tmp_path)
    if mask is not None:
        q = q.filter(mask)
    assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize("n_files", [2, 3])
@pytest.mark.parametrize(
    "blocksize,streaming_engine",
    [
        (1_000, {"executor_options": {"target_partition_size": 1_000}}),
        (10_000, {"executor_options": {"target_partition_size": 10_000}}),
        (1_000_000, {"executor_options": {"target_partition_size": 1_000_000}}),
    ],
    indirect=["streaming_engine"],
)
def test_target_partition_size(tmp_path, df, blocksize, n_files, streaming_engine):
    make_partitioned_source(df, tmp_path, "parquet", n_files=n_files)
    q = pl.scan_parquet(tmp_path)
    assert_gpu_result_equal(q, engine=streaming_engine)

    # Check partitioning (throwaway engine — no cluster/runtime needed)
    _engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"target_partition_size": blocksize},
    )
    qir = Translator(q._ldf.visit(), _engine).translate_ir()
    config_options = ConfigOptions.from_polars_engine(_engine)
    ir, info = lower_ir_graph(
        qir, config_options, collect_statistics(qir, config_options)
    )
    count = info[ir].count
    if blocksize <= 12_000:
        assert count > n_files
    else:
        assert count < n_files
