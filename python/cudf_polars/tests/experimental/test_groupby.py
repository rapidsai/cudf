# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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
        executor_options={"max_rows_per_partition": 4},
    )


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "x": range(150),
            "y": ["cat", "dog", "fish"] * 50,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 30,
        }
    )


@pytest.mark.parametrize("op", ["sum", "mean", "len"])
@pytest.mark.parametrize("keys", [("y",), ("y", "z")])
def test_groupby(df, engine, op, keys):
    q = getattr(df.group_by(*keys), op)()
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("op", ["sum", "mean", "len", "count"])
@pytest.mark.parametrize("keys", [("y",), ("y", "z")])
def test_groupby_agg(df, engine, op, keys):
    q = df.group_by(*keys).agg(getattr(pl.col("x"), op)())
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


def test_groupby_raises(df, engine):
    q = df.group_by("y").median()
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="NotImplementedError",
    ):
        assert_gpu_result_equal(q, engine=engine, check_row_order=False)
