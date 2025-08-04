# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

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


def test_source_statistics(df):
    from cudf_polars.experimental.io import _extract_dataframescan_stats

    row_count = df.collect().height
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "max_rows_per_partition": 1_000,
            "scheduler": DEFAULT_SCHEDULER,
        },
    )
    ir = Translator(df._ldf.visit(), engine).translate_ir()
    column_stats = _extract_dataframescan_stats(ir)

    # Source info is the same for all columns
    source_info = column_stats["x"].source_info
    assert source_info is column_stats["y"].source_info
    assert source_info is column_stats["z"].source_info
    assert source_info.row_count.value == row_count
    assert source_info.row_count.exact

    # Storage stats should not be available
    assert source_info.storage_size("x").value is None

    # Check unique stats
    assert math.isclose(
        source_info.unique_stats("x").count.value, row_count, rel_tol=1e-2
    )
    assert math.isclose(source_info.unique_stats("x").fraction.value, 1.0, abs_tol=1e-2)
    assert not source_info.unique_stats("x").count.exact
    assert math.isclose(source_info.unique_stats("y").count.value, 3, rel_tol=1e-2)
    assert math.isclose(
        source_info.unique_stats("y").fraction.value, 3 / row_count, abs_tol=1e-2
    )
    assert not source_info.unique_stats("y").count.exact
    assert math.isclose(source_info.unique_stats("z").count.value, 5, rel_tol=1e-2)
    assert math.isclose(
        source_info.unique_stats("z").fraction.value, 5 / row_count, abs_tol=1e-2
    )
    assert not source_info.unique_stats("z").count.exact
