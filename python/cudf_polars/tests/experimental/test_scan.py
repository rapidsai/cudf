# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
    assert_gpu_result_equal,
)
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
        executor_options={"cluster": DEFAULT_CLUSTER},
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
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )
    assert_gpu_result_equal(q, engine=engine)

    # Check partitioning
    qir = Translator(q._ldf.visit(), engine).translate_ir()
    ir, info, _ = lower_ir_graph(qir, ConfigOptions.from_polars_engine(engine))
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
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )
    assert_gpu_result_equal(q, engine=engine)
