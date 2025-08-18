# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.testing.asserts import DEFAULT_SCHEDULER, assert_gpu_result_equal
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


@pytest.mark.parametrize("max_rows_per_partition", [1_000, 1_000_000])
def test_parallel_dataframescan(df, max_rows_per_partition):
    total_row_count = len(df.collect())
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": max_rows_per_partition,
            "scheduler": DEFAULT_SCHEDULER,
        },
    )
    assert_gpu_result_equal(df, engine=engine)

    # Check partitioning
    qir = Translator(df._ldf.visit(), engine).translate_ir()
    ir, info = lower_ir_graph(qir, ConfigOptions.from_polars_engine(engine))
    count = info[ir].count
    if max_rows_per_partition < total_row_count:
        assert count > 1
    else:
        assert count == 1


def test_dataframescan_concat(df):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 1_000,
            "scheduler": DEFAULT_SCHEDULER,
        },
    )
    df2 = pl.concat([df, df])
    assert_gpu_result_equal(df2, engine=engine)
