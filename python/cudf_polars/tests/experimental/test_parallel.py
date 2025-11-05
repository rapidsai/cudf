# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle

import pytest

import polars as pl
from polars import GPUEngine
from polars.testing import assert_frame_equal

from cudf_polars import Translator
from cudf_polars.dsl.expressions.base import Col, NamedExpr
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.parallel import (
    get_scheduler,
    lower_ir_graph,
    task_graph,
)
from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
    assert_gpu_result_equal,
)
from cudf_polars.utils.config import ConfigOptions
from cudf_polars.utils.versions import POLARS_VERSION_LT_130


def test_evaluate_streaming():
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5], "c": [5, 6, 7], "d": [7, 9, 8]})
    q = df.select(pl.col("a") - (pl.col("b") + pl.col("c") * 2), pl.col("d")).sort("d")

    expected = q.collect(engine="cpu")
    got_gpu = q.collect(engine=GPUEngine(raise_on_fail=True))
    got_streaming = q.collect(
        engine=GPUEngine(
            raise_on_fail=True,
            executor="streaming",
            executor_options={"cluster": DEFAULT_CLUSTER},
        )
    )
    assert_frame_equal(expected, got_gpu)
    assert_frame_equal(expected, got_streaming)


@pytest.mark.parametrize(
    "agg",
    [
        pl.col("int").max(),
        # Check LiteralColumn serialization
        pl.Series("value", [[4, 5, 6]], dtype=pl.List(pl.Int32)),
    ],
)
def test_pickle_groupby_args(agg):
    df = pl.LazyFrame(
        {
            "key": [1, 1, 1, 2, 3, 1, 4, 6, 7],
            "int": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "float": [7.0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    q = df.group_by(pl.col("key")).agg(agg)
    ir = Translator(q._ldf.visit(), GPUEngine()).translate_ir()
    for node in traversal([ir]):
        pickle.loads(pickle.dumps(node._non_child_args))


def test_pickle_conditional_join_args():
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
    q = left.join_where(right, pl.col("a") < pl.col("a_right"))
    ir = Translator(q._ldf.visit(), GPUEngine()).translate_ir()
    for node in traversal([ir]):
        pickle.loads(pickle.dumps(node._non_child_args))


@pytest.fixture(scope="module")
def engine():
    return pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 2,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
        },
    )


@pytest.mark.parametrize("column", ["a", "b"])
def test_explode_multi(column, engine):
    df = pl.LazyFrame(
        {
            "a": [[1, 2], [3, 4], None],
            "b": [[5, 6], [7, 8], [9, 10]],
            "c": [None, 11, 12],
        }
    )
    q = df.explode(column)
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.parametrize(
    "mapping", [{}, {"b": "c"}, {"b": "a", "a": "b"}, {"a": "c", "b": "d"}]
)
def test_rename_multi(mapping, engine):
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    q = df.rename(mapping)
    assert_gpu_result_equal(q, engine=engine)


def test_preserve_partitioning():
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 2,
            "cluster": DEFAULT_CLUSTER,
            "runtime": DEFAULT_RUNTIME,
            "broadcast_join_limit": 2,
            "unique_fraction": {"a": 1.0},
        },
    )
    left = pl.LazyFrame({"a": [1, 2, 3, 4] * 5, "b": range(20)})
    right = pl.LazyFrame({"a": [3, 4, 5, 6, 7] * 4, "c": range(20)})
    q = (
        left.join(right, on="a")
        .filter(pl.col("a") == 2)
        .group_by(pl.col("a"))
        .mean()
        .select(pl.col("a"), pl.col("c"))
    )
    config_options = ConfigOptions.from_polars_engine(engine)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    ir, partition_info = lower_ir_graph(ir, config_options)
    expect_dtype = ir.schema["a"]
    expect_expr = (NamedExpr("a", Col(expect_dtype, "a")),)
    assert partition_info[ir].partitioned_on == expect_expr
    assert_gpu_result_equal(q, engine=engine)


@pytest.mark.skipif(
    DEFAULT_RUNTIME == "rapidsmpf",
    reason="Uses explicit task graph.",
)
def test_single_cluster():
    # Test that the single cluster clears
    # the cache as tasks are executed.
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 4,
            "cluster": "single",
            "runtime": DEFAULT_RUNTIME,
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
            "xx": range(6),
            "y": [2, 4, 3] * 2,
            "zz": [1, 2] * 3,
        }
    )
    q = left.join(right, on="y").group_by("y").agg(pl.col("zz").mean()).sort(by="y")

    config_options = ConfigOptions.from_polars_engine(engine)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    ir, partition_info = lower_ir_graph(ir, config_options)
    graph, key = task_graph(
        ir,
        partition_info,
        config_options,
    )
    scheduler = get_scheduler(config_options)
    cache = {}
    result = scheduler(graph, key, cache=cache)
    assert_frame_equal(result.to_polars(), q.collect())

    # The cache should only contain the final result
    assert set(cache) == {key}


@pytest.mark.skipif(
    DEFAULT_RUNTIME == "rapidsmpf",
    reason="Uses explicit task graph.",
)
def test_task_graph_is_pickle_serializable(engine):
    # Dask will fall back to using cloudpickle to serialize the task graph if
    # necessary. We'd like to avoid that, since cloudpickle serialization /
    # deserialization is typically slower than pickle.

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
    q = left.join(right, on="a").group_by("a").agg(pl.col("c").sum())

    config_options = ConfigOptions.from_polars_engine(engine)
    ir = Translator(q._ldf.visit(), engine).translate_ir()
    ir, partition_info = lower_ir_graph(ir, config_options)
    graph, _ = task_graph(
        ir,
        partition_info,
        config_options,
    )

    pickle.loads(pickle.dumps(graph))  # no exception


def test_rename_concat(engine: pl.GPUEngine) -> None:
    # https://github.com/rapidsai/cudf/pull/19121#issuecomment-2959305678
    q = pl.concat(
        [
            pl.LazyFrame({"a": [1, 2, 3]}).rename({"a": "A"}),
            pl.LazyFrame({"a": [4, 5, 6]}).rename({"a": "A"}),
        ]
    )
    assert_gpu_result_equal(q, engine=engine)


def test_fallback_on_concat_zlice(engine: pl.GPUEngine) -> None:
    q = pl.concat(
        [
            pl.LazyFrame({"a": [1, 2]}),
            pl.LazyFrame({"a": [3, 4]}),
            pl.LazyFrame({"a": [5, 6]}),
        ]
    ).tail(1)

    if POLARS_VERSION_LT_130:
        with pytest.raises(
            pl.exceptions.ComputeError,
            match="This slice not supported for multiple partitions.",
        ):
            assert_gpu_result_equal(q, engine=engine)
    else:
        with pytest.raises(
            UserWarning, match="This slice not supported for multiple partitions."
        ):
            assert_gpu_result_equal(q, engine=engine)
