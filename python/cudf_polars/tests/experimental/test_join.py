# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.experimental.shuffle import Shuffle
from cudf_polars.testing.asserts import DEFAULT_SCHEDULER, assert_gpu_result_equal
from cudf_polars.utils.config import ConfigOptions


@pytest.fixture(scope="module")
def left():
    return pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )


@pytest.fixture(scope="module")
def right():
    return pl.LazyFrame(
        {
            "xx": range(9),
            "y": [2, 4, 3] * 3,
            "zz": [1, 2, 3] * 3,
        }
    )


@pytest.mark.parametrize("how", ["inner", "left", "right", "full", "semi", "anti"])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("max_rows_per_partition", [1, 5, 10, 15])
@pytest.mark.parametrize("broadcast_join_limit", [1, 16])
def test_join(left, right, how, reverse, max_rows_per_partition, broadcast_join_limit):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "scheduler": DEFAULT_SCHEDULER,
            "max_rows_per_partition": max_rows_per_partition,
            "broadcast_join_limit": broadcast_join_limit,
            "shuffle_method": "tasks",
        },
    )
    if reverse:
        left, right = right, left

    q = left.join(right, on="y", how=how)

    assert_gpu_result_equal(q, engine=engine, check_row_order=False)

    # Join again on the same key.
    # (covers code path that avoids redundant shuffles)
    if how in ("inner", "left", "right"):
        right2 = pl.LazyFrame(
            {
                "xxx": range(6),
                "yyy": [2, 4, 3] * 2,
                "zzz": [3, 4] * 3,
            }
        )
        q2 = q.join(right2, left_on="y", right_on="yyy", how=how)
        assert_gpu_result_equal(q2, engine=engine, check_row_order=False)


@pytest.mark.parametrize("broadcast_join_limit", [1, 2, 3, 4])
def test_broadcast_join_limit(left, right, broadcast_join_limit):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "broadcast_join_limit": broadcast_join_limit,
            "scheduler": DEFAULT_SCHEDULER,
            "shuffle_method": "tasks",
        },
    )
    left = pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )
    right = pl.LazyFrame(
        {
            "xx": range(9),
            "y": [2, 4, 3] * 3,
            "zz": [1, 2, 3] * 3,
        }
    )

    q = left.join(right, on="y", how="inner")
    shuffle_nodes = [
        type(node)
        for node in lower_ir_graph(
            Translator(q._ldf.visit(), engine).translate_ir(),
            ConfigOptions.from_polars_engine(engine),
        )[1]
        if isinstance(node, Shuffle)
    ]

    # NOTE: Expect small table to have 3 partitions (9 / 3).
    # Therefore, we will get a shuffle-based join if our
    # "broadcast_join_limit" config is less than 3.
    if broadcast_join_limit < 3:
        # Expect shuffle-based join
        assert len(shuffle_nodes) == 2
    else:
        # Expect broadcast join
        assert len(shuffle_nodes) == 0


def test_join_then_shuffle(left, right):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "scheduler": DEFAULT_SCHEDULER,
            "max_rows_per_partition": 2,
            "broadcast_join_limit": 1,
        },
    )
    q = left.join(right, on="y", how="inner").select(
        pl.col("x").sum(),
        pl.col("xx").mean(),
        pl.col("y").n_unique(),
        (pl.col("y") * pl.col("y")).n_unique().alias("y2"),
    )

    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("max_rows_per_partition", [3, 9])
@pytest.mark.parametrize("reverse", [True, False])
def test_join_conditional(reverse, max_rows_per_partition):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": max_rows_per_partition,
            "scheduler": DEFAULT_SCHEDULER,
            "fallback_mode": "silent",
        },
    )
    left = pl.LazyFrame({"x": range(15), "y": [1, 2, 3] * 5})
    right = pl.LazyFrame({"xx": range(9), "yy": [2, 4, 3] * 3})
    if reverse:
        left, right = right, left
    q = left.join_where(right, pl.col("y") < pl.col("yy"))
    assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("zlice", [(0, 2), (2, 2), (-2, None)])
def test_join_and_slice(zlice):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "broadcast_join_limit": 100,
            "scheduler": DEFAULT_SCHEDULER,
            "shuffle_method": "tasks",
            "fallback_mode": "warn" if zlice[0] == 0 else "silent",
        },
    )
    left = pl.LazyFrame(
        {
            "a": [1, 2, 3, 1, None],
            "b": [1, 2, 3, 4, 5],
            "c": [2, 3, 4, 5, 6],
        }
    )
    right = pl.LazyFrame(
        {
            "a": [1, 4, 3, 7, None, None, 1],
            "c": [2, 3, 4, 5, 6, 7, 8],
            "d": [6, None, 7, 8, -1, 2, 4],
        }
    )
    q = left.join(right, on="a", how="inner").slice(*zlice)
    # Check that we get the correct row count
    # See: https://github.com/rapidsai/cudf/issues/19153
    assert q.collect(engine=engine).height == q.collect().height

    # Need sort to match order after a join
    q = left.join(right, on="a", how="inner").sort(pl.col("a")).slice(*zlice)
    assert_gpu_result_equal(q, engine=engine)
