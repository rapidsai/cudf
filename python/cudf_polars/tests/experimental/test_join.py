# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture(scope="module")
def engine():
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="dask-experimental",
    )


@pytest.fixture(scope="module")
def module_tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="module")
def df(module_tmp_dir):
    pdf = pl.DataFrame(
        {
            "x": range(30),
            "y": ["cat", "dog", "fish"] * 10,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 6,
        }
    )
    n_files = 3
    n_rows = len(pdf)
    stride = int(n_rows / n_files)
    for i in range(n_files):
        offset = stride * i
        part = pdf.slice(offset, stride)
        part.write_parquet(module_tmp_dir / f"part.{i}.parquet")
    return pl.scan_parquet(module_tmp_dir)


@pytest.mark.parametrize("how", ["inner", "left", "right"])
def test_parallel_join(df, engine, how):
    other = pl.LazyFrame(
        {
            "y": ["dog", "bird", "fish", "snake"] * 30,
            "zz": [1, 2, 3] * 40,
        }
    )
    q = df.join(other, on="y", how=how)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
