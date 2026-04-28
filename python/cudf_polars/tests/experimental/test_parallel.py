# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
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
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.config import ConfigOptions


@pytest.mark.parametrize("column", ["a", "b"])
def test_explode_multi(column, streaming_engine):
    df = pl.LazyFrame(
        {
            "a": [[1, 2], [3, 4], None],
            "b": [[5, 6], [7, 8], [9, 10]],
            "c": [None, 11, 12],
        }
    )
    q = df.explode(column)
    assert_gpu_result_equal(q, engine=streaming_engine)


@pytest.mark.parametrize(
    "mapping", [{}, {"b": "c"}, {"b": "a", "a": "b"}, {"a": "c", "b": "d"}]
)
def test_rename_multi(mapping, streaming_engine):
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    q = df.rename(mapping)
    assert_gpu_result_equal(q, engine=streaming_engine)


def test_rename_concat(streaming_engine) -> None:
    # https://github.com/rapidsai/cudf/pull/19121#issuecomment-2959305678
    q = pl.concat(
        [
            pl.LazyFrame({"a": [1, 2, 3]}).rename({"a": "A"}),
            pl.LazyFrame({"a": [4, 5, 6]}).rename({"a": "A"}),
        ]
    )
    assert_gpu_result_equal(q, engine=streaming_engine)


def test_fallback_on_concat_zlice(streaming_engine) -> None:
    q = pl.concat(
        [
            pl.LazyFrame({"a": [1, 2]}),
            pl.LazyFrame({"a": [3, 4]}),
            pl.LazyFrame({"a": [5, 6]}),
        ]
    ).tail(1)

    with pytest.raises(
        UserWarning, match="This slice not supported for multiple partitions."
    ):
        assert_gpu_result_equal(q, engine=streaming_engine)


# ---------------------------------------------------------------------------
# Tests migrated from tests/experimental/test_parallel.py
# ---------------------------------------------------------------------------


def test_evaluate_streaming(streaming_engine):
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5], "c": [5, 6, 7], "d": [7, 9, 8]})
    q = df.select(pl.col("a") - (pl.col("b") + pl.col("c") * 2), pl.col("d")).sort("d")

    expected = q.collect(engine="cpu")
    got_gpu = q.collect(engine=pl.GPUEngine(raise_on_fail=True))
    got_streaming = q.collect(engine=streaming_engine)
    assert_frame_equal(expected, got_gpu)
    assert_frame_equal(expected, got_streaming)


# ---------------------------------------------------------------------------
# Tests migrated from tests/experimental/test_parallel.py (round 3)
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize(
    "streaming_engine",
    [
        {
            "executor_options": {
                "max_rows_per_partition": 2,
                "broadcast_join_limit": 2,
                "unique_fraction": {"a": 1.0},
            }
        }
    ],
    indirect=True,
)
def test_preserve_partitioning(streaming_engine):
    left = pl.LazyFrame({"a": [1, 2, 3, 4] * 5, "b": range(20)})
    right = pl.LazyFrame({"a": [3, 4, 5, 6, 7] * 4, "c": range(20)})
    q = (
        left.join(right, on="a")
        .filter(pl.col("a") == 2)
        .group_by(pl.col("a"))
        .mean()
        .select(pl.col("a"), pl.col("c"))
    )
    _engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 2,
            "broadcast_join_limit": 2,
            "unique_fraction": {"a": 1.0},
        },
    )
    config_options = ConfigOptions.from_polars_engine(_engine)
    ir = Translator(q._ldf.visit(), _engine).translate_ir()
    ir, partition_info, _ = lower_ir_graph(ir, config_options)
    expect_dtype = ir.schema["a"]
    expect_expr = (NamedExpr("a", Col(expect_dtype, "a")),)
    assert partition_info[ir].partitioned_on == expect_expr
    assert_gpu_result_equal(q, engine=streaming_engine)
