# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture(scope="module")
def df():
    return pl.DataFrame(
        {
            "x": range(3_000),
            "y": ["cat", "dog", "fish"] * 1_000,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 600,
        }
    )


def make_source(df, path, fmt, n_files=3):
    n_rows = len(df)
    stride = int(n_rows / n_files)
    for i in range(n_files):
        offset = stride * i
        part = df.slice(offset, stride)
        if fmt == "csv":
            part.write_csv(path / f"part.{i}.csv")
        elif fmt == "ndjson":
            part.write_ndjson(path / f"part.{i}.ndjson")
        else:
            part.write_parquet(
                path / f"part.{i}.parquet",
                row_group_size=int(stride / 2),
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
    make_source(df, tmp_path, fmt)
    q = scan_fn(tmp_path)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="dask-experimental",
    )
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("blocksize", [1_000, 10_000, 1_000_000])
def test_parquet_blocksize(tmp_path, df, blocksize):
    n_files = 3
    make_source(df, tmp_path, "parquet", n_files)
    q = pl.scan_parquet(tmp_path)
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="dask-experimental",
        executor_options={"parquet_blocksize": blocksize},
    )
    assert_gpu_result_equal(q, engine=engine)

    # Check partitioning
    qir = Translator(q._ldf.visit(), engine).translate_ir()
    ir, info = lower_ir_graph(qir)
    count = info[ir].count
    if blocksize <= 12_000:
        assert count > n_files
    else:
        assert count < n_files
