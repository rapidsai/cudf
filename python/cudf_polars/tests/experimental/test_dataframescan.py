# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.dsl.traversal import traversal
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.experimental.statistics import collect_statistics
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.config import ConfigOptions


def _assert_stable_ids_match(orig, loaded) -> None:
    for a, b in zip(traversal([orig]), traversal([loaded]), strict=True):
        assert a.get_stable_id() == b.get_stable_id()


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "x": range(30_000),
            "y": [1, 2, 3] * 10_000,
            "z": [1.0, 2.0, 3.0, 4.0, 5.0] * 6_000,
        }
    )


@pytest.mark.parametrize(
    "max_rows_per_partition,streaming_engine",
    [
        (1_000, {"executor_options": {"max_rows_per_partition": 1_000}}),
        (1_000_000, {"executor_options": {"max_rows_per_partition": 1_000_000}}),
    ],
    indirect=["streaming_engine"],
)
def test_parallel_dataframescan(df, max_rows_per_partition, streaming_engine):
    total_row_count = len(df.collect())
    assert_gpu_result_equal(df, engine=streaming_engine)

    # Check partitioning (throwaway engine — no cluster/runtime needed)
    _engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"max_rows_per_partition": max_rows_per_partition},
    )
    qir = Translator(df._ldf.visit(), _engine).translate_ir()
    config_options = ConfigOptions.from_polars_engine(_engine)
    ir, info = lower_ir_graph(
        qir, config_options, collect_statistics(qir, config_options)
    )
    count = info[ir].count
    if max_rows_per_partition < total_row_count:
        assert count > 1
    else:
        assert count == 1


@pytest.mark.parametrize(
    "streaming_engine",
    [{"executor_options": {"max_rows_per_partition": 1_000}}],
    indirect=True,
)
def test_dataframescan_concat(df, streaming_engine):
    df2 = pl.concat([df, df])
    assert_gpu_result_equal(df2, engine=streaming_engine)


def test_join_in_memory_lazy_stable_id_pickle():
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"max_rows_per_partition": 1_000},
    )
    left = pl.LazyFrame({"k": [1, 2, 3], "x": [10, 20, 30]}).collect().lazy()
    right = pl.LazyFrame({"k": [2, 3, 4], "y": [1, 2, 3]}).collect().lazy()
    qir = Translator(left.join(right, on="k")._ldf.visit(), engine).translate_ir()
    config_options = ConfigOptions.from_polars_engine(engine)
    ir, _ = lower_ir_graph(qir, config_options, collect_statistics(qir, config_options))
    _assert_stable_ids_match(ir, pickle.loads(pickle.dumps(ir)))


def test_dataframescan_pickle(df):
    _engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"max_rows_per_partition": 1_000},
    )
    qir = Translator(df._ldf.visit(), _engine).translate_ir()
    config_options = ConfigOptions.from_polars_engine(_engine)
    ir, _ = lower_ir_graph(qir, config_options, collect_statistics(qir, config_options))

    # Pickle and unpickle the IR (which contains DataFrameScan)
    pickled = pickle.dumps(ir)
    unpickled_ir = pickle.loads(pickled)

    # Verify the unpickled IR is equivalent
    assert type(unpickled_ir) is type(ir)
    assert unpickled_ir.schema == ir.schema
    _assert_stable_ids_match(ir, unpickled_ir)
