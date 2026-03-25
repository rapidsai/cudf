# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.testing.asserts import assert_gpu_result_equal
from cudf_polars.utils.config import ConfigOptions


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
    "max_rows_per_partition,engine",
    [
        (1_000, {"executor_options": {"max_rows_per_partition": 1_000}}),
        (1_000_000, {"executor_options": {"max_rows_per_partition": 1_000_000}}),
    ],
    indirect=["engine"],
)
def test_parallel_dataframescan(df, max_rows_per_partition, engine):
    total_row_count = len(df.collect())
    assert_gpu_result_equal(df, engine=engine)

    # Check partitioning (throwaway engine — no cluster/runtime needed)
    _engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"max_rows_per_partition": max_rows_per_partition},
    )
    qir = Translator(df._ldf.visit(), _engine).translate_ir()
    ir, info, _ = lower_ir_graph(qir, ConfigOptions.from_polars_engine(_engine))
    count = info[ir].count
    if max_rows_per_partition < total_row_count:
        assert count > 1
    else:
        assert count == 1


@pytest.mark.parametrize(
    "engine",
    [{"executor_options": {"max_rows_per_partition": 1_000}}],
    indirect=True,
)
def test_dataframescan_concat(df, engine):
    df2 = pl.concat([df, df])
    assert_gpu_result_equal(df2, engine=engine)


def test_dataframescan_pickle(df):
    _engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={"max_rows_per_partition": 1_000},
    )
    qir = Translator(df._ldf.visit(), _engine).translate_ir()
    ir, _, _ = lower_ir_graph(qir, ConfigOptions.from_polars_engine(_engine))

    # Pickle and unpickle the IR (which contains DataFrameScan)
    pickled = pickle.dumps(ir)
    unpickled_ir = pickle.loads(pickled)

    # Verify the unpickled IR is equivalent
    assert type(unpickled_ir) is type(ir)
    assert unpickled_ir.schema == ir.schema
