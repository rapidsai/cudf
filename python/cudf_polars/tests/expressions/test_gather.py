# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.versions import POLARS_VERSION_LT_130


def test_gather():
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [0, 3, 1, 5, 6, 1, 0],
        }
    )

    query = ldf.select(pl.col("a").gather(pl.col("b")))
    assert_gpu_result_equal(query)


def test_gather_with_nulls():
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [0, None, 1, None, 6, 1, 0],
        }
    )

    query = ldf.select(pl.col("a").gather(pl.col("b")))

    assert_gpu_result_equal(query)


@pytest.mark.parametrize("negative", [False, True])
def test_gather_out_of_bounds(negative):
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [0, -10 if negative else 10, 1, 2, 6, 1, 0],
        }
    )

    query = ldf.select(pl.col("a").gather(pl.col("b")))

    if POLARS_VERSION_LT_130:
        with pytest.raises(pl.exceptions.ComputeError):
            query.collect(engine="gpu")
    else:
        with pytest.raises(ValueError, match="gather indices are out of bounds"):
            query.collect(engine="gpu")
