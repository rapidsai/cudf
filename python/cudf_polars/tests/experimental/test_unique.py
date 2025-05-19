# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars.testing.asserts import DEFAULT_SCHEDULER, assert_gpu_result_equal


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "x": range(150),
            "y": list(range(30)) * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 30,
        }
    )


@pytest.mark.parametrize("subset", [None, ("y",), ("y", "z")])
@pytest.mark.parametrize("keep", ["first", "last", "any", "none"])
@pytest.mark.parametrize("maintain_order", [True, False])
@pytest.mark.parametrize("unique_fraction", [{}, {"y": 0.7}])
def test_unique(df, keep, subset, maintain_order, unique_fraction):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 50,
            "scheduler": DEFAULT_SCHEDULER,
            "unique_fraction": unique_fraction,
            "fallback_mode": "silent",
        },
    )

    q = df.unique(subset=subset, keep=keep, maintain_order=maintain_order)
    check_row_order = maintain_order
    if keep == "any" and subset:
        q = q.select(*(pl.col(col) for col in subset))
        check_row_order = False

    assert_gpu_result_equal(q, engine=engine, check_row_order=check_row_order)


def test_unique_fallback(df):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 50,
            "scheduler": DEFAULT_SCHEDULER,
            "unique_fraction": {"y": 1.0},
            "fallback_mode": "raise",
        },
    )
    q = df.unique(keep="first", maintain_order=True)
    with pytest.raises(pl.exceptions.ComputeError, match="Unsupported unique options"):
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize("maintain_order", [True, False])
@pytest.mark.parametrize("unique_fraction", [{}, {"y": 0.5}])
def test_unique_select(df, maintain_order, unique_fraction):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 4,
            "scheduler": DEFAULT_SCHEDULER,
            "unique_fraction": unique_fraction,
            "fallback_mode": "silent",
        },
    )

    q = df.select(pl.col("y").unique(maintain_order=maintain_order))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("keep", ["first", "last", "any"])
@pytest.mark.parametrize("zlice", ["head", "tail"])
def test_unique_head_tail(keep, zlice):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 4,
            "scheduler": DEFAULT_SCHEDULER,
        },
    )
    data = [0, 1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8, 9, 10]
    df = pl.LazyFrame({"x": data})
    q = df.unique(subset=None, keep=keep, maintain_order=True)
    expect = pl.LazyFrame({"x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    # Cannot use assert_gpu_result_equal until CPU bug is fixed
    # See: https://github.com/pola-rs/polars/issues/22470
    assert_frame_equal(
        getattr(q, zlice)().collect(engine=engine),
        getattr(expect, zlice)().collect(),
    )
