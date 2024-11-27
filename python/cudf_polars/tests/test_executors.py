# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("executor", [None, "pylibcudf", "dask-experimental"])
def test_executor_basics(executor):
    if executor == "dask-experimental":
        pytest.importorskip("dask")

    df = pl.LazyFrame(
        {
            "a": pl.Series([[1, 2], [3]], dtype=pl.List(pl.Int8())),
            "b": pl.Series([[1], [2]], dtype=pl.List(pl.UInt16())),
            "c": pl.Series(
                [
                    [["1", "2", "3"], ["4", "567"]],
                    [["8", "9"], []],
                ],
                dtype=pl.List(pl.List(pl.String())),
            ),
            "d": pl.Series([[[1, 2]], []], dtype=pl.List(pl.List(pl.UInt16()))),
        }
    )

    assert_gpu_result_equal(df, executor=executor)


def test_cudf_cache_evaluate():
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    ).lazy()
    ldf2 = ldf.select((pl.col("a") + pl.col("b")).alias("c"), pl.col("a"))
    query = pl.concat([ldf, ldf2], how="diagonal")
    assert_gpu_result_equal(query, executor="pylibcudf")


def test_dask_experimental_map_function_get_hashable():
    df = pl.LazyFrame(
        {
            "a": pl.Series([11, 12, 13], dtype=pl.UInt16),
            "b": pl.Series([1, 3, 5], dtype=pl.Int16),
            "c": pl.Series([2, 4, 6], dtype=pl.Float32),
            "d": ["a", "b", "c"],
        }
    )
    q = df.unpivot(index="d")
    assert_gpu_result_equal(q, executor="dask-experimental")


def test_unknown_executor():
    df = pl.LazyFrame({})

    with pytest.raises(
        pl.exceptions.ComputeError,
        match="ValueError: Unknown executor 'unknown-executor'",
    ):
        assert_gpu_result_equal(df, executor="unknown-executor")


@pytest.mark.parametrize("executor", [None, "pylibcudf", "dask-experimental"])
def test_unknown_executor_options(executor):
    df = pl.LazyFrame({})

    with pytest.raises(
        pl.exceptions.ComputeError,
        match="Unsupported executor_options",
    ):
        df.collect(
            engine=pl.GPUEngine(
                executor=executor,
                executor_options={"foo": None},
            )
        )
