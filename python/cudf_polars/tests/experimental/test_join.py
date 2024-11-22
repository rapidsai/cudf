# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("how", ["inner"])
@pytest.mark.parametrize("num_rows_threshold", [5, 10, 15])
def test_parallel_join(how, num_rows_threshold):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="dask-experimental",
        parallel_options={"num_rows_threshold": num_rows_threshold},
    )
    left = pl.LazyFrame(
        {
            "x": range(15),
            "y": ["cat", "dog", "fish"] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    right = pl.LazyFrame(
        {
            "xx": range(6),
            "y": ["dog", "bird", "fish"] * 2,
            "zz": [1, 2] * 3,
        }
    )
    q = left.join(right, on="y", how=how)
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
