# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("how", ["inner", "left", "right", "full", "semi", "anti"])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("max_rows_per_partition", [1, 5, 10, 15])
def test_join(how, reverse, max_rows_per_partition):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="dask-experimental",
        executor_options={"max_rows_per_partition": max_rows_per_partition},
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
    if reverse:
        left, right = right, left

    q = left.join(right, on="y", how=how)

    assert_gpu_result_equal(q, engine=engine, check_row_order=False)
