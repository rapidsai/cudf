# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle

import pytest

import polars as pl
from polars import GPUEngine
from polars.testing import assert_frame_equal

from cudf_polars import Translator
from cudf_polars.dsl.traversal import traversal


def test_evaluate_dask():
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5], "c": [5, 6, 7], "d": [7, 9, 8]})
    q = df.select(pl.col("a") - (pl.col("b") + pl.col("c") * 2), pl.col("d")).sort("d")

    expected = q.collect(engine="cpu")
    got_gpu = q.collect(engine=GPUEngine(raise_on_fail=True))
    got_dask = q.collect(
        engine=GPUEngine(raise_on_fail=True, executor="dask-experimental")
    )
    assert_frame_equal(expected, got_gpu)
    assert_frame_equal(expected, got_dask)


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
