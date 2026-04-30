# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle

import pytest

import polars as pl
from polars.testing import assert_frame_equal

from cudf_polars import Translator
from cudf_polars.experimental.parallel import (
    get_scheduler,
    lower_ir_graph,
    task_graph,
)
from cudf_polars.experimental.statistics import collect_statistics
from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    DEFAULT_RUNTIME,
)
from cudf_polars.utils.config import ConfigOptions


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
    ir, partition_info = lower_ir_graph(
        ir, config_options, collect_statistics(ir, config_options)
    )
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
    ir, partition_info = lower_ir_graph(
        ir, config_options, collect_statistics(ir, config_options)
    )
    graph, _ = task_graph(
        ir,
        partition_info,
        config_options,
    )

    pickle.loads(pickle.dumps(graph))  # no exception
