# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dynamic join path in join_actor (including Right and Full joins)."""

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.dsl.ir import Cache, Join
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.base import PartitionInfo
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.experimental.rapidsmpf.frontend.options import StreamingOptions
from cudf_polars.experimental.rapidsmpf.join import _use_pwise_join
from cudf_polars.experimental.shuffle import Shuffle
from cudf_polars.experimental.statistics import collect_statistics
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.config import ConfigOptions, StreamingExecutor


@pytest.fixture
def left():
    return pl.LazyFrame(
        {
            "x": range(15),
            "y": [1, 2, 3] * 5,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 3,
        }
    )


@pytest.fixture
def right():
    return pl.LazyFrame(
        {
            "xx": range(9),
            "y": [2, 4, 3] * 3,
            "zz": [1, 2, 3] * 3,
        }
    )


@pytest.mark.parametrize("how", ["inner", "left", "right", "full"])
@pytest.mark.parametrize(
    "options",
    [
        StreamingOptions(max_rows_per_partition=3, broadcast_join_limit=2),
        StreamingOptions(max_rows_per_partition=5, broadcast_join_limit=2),
    ],
)
def test_dynamic_join_how(left, right, streaming_engine_factory, options, how):
    """Dynamic join path: all join types including Right and Full."""
    streaming_engine = streaming_engine_factory(options)
    q = left.join(right, on="y", how=how)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize("how", ["right", "full"])
def test_dynamic_join_right_full_reverse(left, right, streaming_engine_factory, how):
    """Dynamic join path: Right/Full with reversed left/right (stress ordering)."""
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=3, broadcast_join_limit=2),
    )
    # Reverse so "right" frame is larger; exercises right-side preservation
    q = right.join(left, on="y", how=how)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


# ---------------------------------------------------------------------------
# Tests migrated from tests/experimental/test_join.py
# ---------------------------------------------------------------------------


def test_join_then_shuffle(left, right, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=2, broadcast_join_limit=1),
    )
    q = left.join(right, on="y", how="inner").select(
        pl.col("x").sum(),
        pl.col("xx").mean(),
        pl.col("y").n_unique(),
        (pl.col("y") * pl.col("y")).n_unique().alias("y2"),
    )
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("max_rows_per_partition", [3, 9])
def test_join_conditional(reverse, max_rows_per_partition, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=max_rows_per_partition,
            fallback_mode="warn",
            dynamic_planning=None,
        ),
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
            assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)
    else:
        assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


# ---------------------------------------------------------------------------
# Tests migrated from tests/experimental/test_join.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["inner", "left", "right", "full", "semi", "anti"])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize(
    "options",
    [
        StreamingOptions(max_rows_per_partition=1, broadcast_join_limit=1),
        StreamingOptions(max_rows_per_partition=1, broadcast_join_limit=16),
        StreamingOptions(max_rows_per_partition=5, broadcast_join_limit=1),
        StreamingOptions(max_rows_per_partition=5, broadcast_join_limit=16),
        StreamingOptions(max_rows_per_partition=10, broadcast_join_limit=1),
        StreamingOptions(max_rows_per_partition=10, broadcast_join_limit=16),
        StreamingOptions(max_rows_per_partition=15, broadcast_join_limit=1),
        StreamingOptions(max_rows_per_partition=15, broadcast_join_limit=16),
    ],
)
def test_join(left, right, how, reverse, streaming_engine_factory, options):
    streaming_engine = streaming_engine_factory(options)
    if reverse:
        left, right = right, left

    q = left.join(right, on="y", how=how)

    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)

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
        assert_gpu_result_equal(q2, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize("zlice", [(0, 2), (2, 2), (-2, None)])
def test_join_and_slice(zlice, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=3,
            broadcast_join_limit=100,
            fallback_mode="warn",
        ),
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
            assert q.collect(engine=streaming_engine).height == q.collect().height
    else:
        assert q.collect(engine=streaming_engine).height == q.collect().height

    # Need sort to match order after a join
    q = left.join(right, on="a", how="inner").sort(pl.col("a")).slice(*zlice)
    if zlice == (2, 2):
        with pytest.warns(
            UserWarning,
            match="This slice not supported for multiple partitions.",
        ):
            assert_gpu_result_equal(q, engine=streaming_engine)
    else:
        assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize("how", ["inner", "semi", "left", "right"])
def test_bloom_filter_join(how, streaming_engine_factory):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=2,
            broadcast_join_limit=1,
            target_partition_size=10,
        ),
    )
    dim = pl.LazyFrame({"key": range(10), "val": range(10)})
    fact = pl.LazyFrame({"key": range(200), "data": range(200)})
    left, right = (dim, fact) if how == "right" else (fact, dim)
    q = left.join(right, on="key", how=how)
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)


@pytest.mark.parametrize(
    "maintain_order", ["left_right", "right_left", "left", "right"]
)
def test_join_maintain_order_fallback_streaming(
    left, right, maintain_order, streaming_engine_factory
):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=3,
            broadcast_join_limit=1,
            fallback_mode="warn",
        ),
    )
    q = left.join(right, on="y", how="inner", maintain_order=maintain_order)

    with pytest.warns(
        UserWarning,
        match=r"Join\(maintain_order=.*\) not supported for multiple partitions\.",
    ):
        assert_gpu_result_equal(q, engine=streaming_engine)


# ---------------------------------------------------------------------------
# Tests migrated from tests/experimental/test_join.py (IR inspection)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("broadcast_join_limit", [1, 2, 3, 4])
def test_broadcast_join_limit(left, right, broadcast_join_limit):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "broadcast_join_limit": broadcast_join_limit,
            "dynamic_planning": None,  # Requires static planning
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
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    config_options = ConfigOptions.from_polars_engine(engine)
    shuffle_nodes = [
        type(node)
        for node in lower_ir_graph(
            ir,
            config_options,
            collect_statistics(ir, config_options),
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


def test_cache_preserves_partitioning_join():
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 3,
            "dynamic_planning": None,  # Requires static planning
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
    lowered_ir, partition_info = lower_ir_graph(
        ir, config_options, collect_statistics(ir, config_options)
    )

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


def test_dynamic_planning_skips_compile_time_partition_wise_join():
    lf = pl.LazyFrame({"y": [1, 2, 3]})
    q = lf.join(lf, on="y", how="inner")
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"dynamic_planning": {}},
    )
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    join_ir = next(n for n in traversal([ir]) if isinstance(n, Join))
    left_ir, right_ir = join_ir.children
    executor = StreamingExecutor(dynamic_planning={})
    partition_info = {
        join_ir: PartitionInfo(1, partitioned_on=()),
        left_ir: PartitionInfo(1, partitioned_on=()),
        right_ir: PartitionInfo(1, partitioned_on=()),
    }
    assert not _use_pwise_join(executor, partition_info, join_ir)
