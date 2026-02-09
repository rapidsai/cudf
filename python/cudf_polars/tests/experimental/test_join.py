# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.dsl.ir import Cache
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.experimental.shuffle import Shuffle
from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
    assert_gpu_result_equal,
)
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
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "max_rows_per_partition": max_rows_per_partition,
            "broadcast_join_limit": broadcast_join_limit,
            "shuffle_method": DEFAULT_RUNTIME,  # Names coincide
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
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "shuffle_method": DEFAULT_RUNTIME,  # Names coincide
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
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
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
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "fallback_mode": "warn",
        },
    )
    left = pl.LazyFrame({"x": range(15), "y": [1, 2, 3] * 5})
    right = pl.LazyFrame({"xx": range(9), "yy": [2, 4, 3] * 3})
    if reverse:
        left, right = right, left
    q = left.join_where(right, pl.col("y") < pl.col("yy"))
    if max_rows_per_partition == 3:
        with pytest.warns(
            UserWarning, match="ConditionalJoin not supported for multiple partitions."
        ):
            assert_gpu_result_equal(q, engine=engine, check_row_order=False)
    else:
        assert_gpu_result_equal(q, engine=engine, check_row_order=False)


@pytest.mark.parametrize("zlice", [(0, 2), (2, 2), (-2, None)])
def test_join_and_slice(zlice):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "broadcast_join_limit": 100,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "shuffle_method": DEFAULT_RUNTIME,  # Names coincide
            "fallback_mode": "warn",
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
    if zlice in {(2, 2), (-2, None)}:
        with pytest.warns(
            UserWarning, match="This slice not supported for multiple partitions."
        ):
            assert q.collect(engine=engine).height == q.collect().height
    else:
        assert q.collect(engine=engine).height == q.collect().height

    # Need sort to match order after a join
    q = left.join(right, on="a", how="inner").sort(pl.col("a")).slice(*zlice)
    if zlice == (2, 2):
        with pytest.warns(
            UserWarning,
            match="does not support a multi-partition slice with an offset.",
        ):
            assert_gpu_result_equal(q, engine=engine)
    else:
        assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "maintain_order", ["left_right", "right_left", "left", "right"]
)
def test_join_maintain_order_fallback_streaming(left, right, maintain_order):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "max_rows_per_partition": 3,
            "broadcast_join_limit": 1,
            "shuffle_method": DEFAULT_RUNTIME,  # Names coincide
            "fallback_mode": "warn",
        },
    )

    q = left.join(right, on="y", how="inner", maintain_order=maintain_order)

    with pytest.warns(
        UserWarning,
        match=r"Join\(maintain_order=.*\) not supported for multiple partitions\.",
    ):
        assert_gpu_result_equal(q, engine=engine)


def test_cache_preserves_partitioning_join():
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )

    left = pl.LazyFrame({"key": list(range(20)) * 5, "val_a": range(100)})
    right = pl.LazyFrame({"key": list(range(20)) * 5, "val_b": range(100)})
    joined = left.join(right, on="key")

    # Use joined result twice to trigger Cache (CSE)
    q = pl.concat(
        [
            joined.group_by("key").agg(pl.col("val_a").sum()),
            joined.group_by("key").agg(pl.col("val_b").sum()),
        ]
    )

    config_options = ConfigOptions.from_polars_engine(engine)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    lowered_ir, partition_info, _ = lower_ir_graph(ir, config_options)

    # Cache should preserve partitioning on 'key'
    cache_partitioning = [
        [ne.name for ne in partition_info[node].partitioned_on]
        for node in traversal([lowered_ir])
        if isinstance(node, Cache)
    ]
    assert cache_partitioning == [["key"]], (
        f"Cache should preserve partitioning on 'key', got {cache_partitioning}"
    )

    # Only 2 shuffles needed (for join sides, not for groupby)
    num_shuffles = sum(
        1 for node in traversal([lowered_ir]) if isinstance(node, Shuffle)
    )
    assert num_shuffles == 2, f"Expected 2 shuffles, got {num_shuffles}"
