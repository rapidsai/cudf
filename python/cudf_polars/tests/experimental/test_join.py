# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import Translator
from cudf_polars.experimental.parallel import lower_ir_graph
from cudf_polars.experimental.shuffle import Shuffle
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("how", ["inner", "left", "right", "full", "semi", "anti"])
@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("max_rows_per_partition", [1, 5, 10, 15])
@pytest.mark.parametrize("broadcast_join_limit", [1, 16])
def test_join(how, reverse, max_rows_per_partition, broadcast_join_limit):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="dask-experimental",
        executor_options={
            "max_rows_per_partition": max_rows_per_partition,
            "broadcast_join_limit": broadcast_join_limit,
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
def test_broadcast_join_limit(broadcast_join_limit):
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="dask-experimental",
        executor_options={
            "max_rows_per_partition": 3,
            "broadcast_join_limit": broadcast_join_limit,
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
        for node in lower_ir_graph(Translator(q._ldf.visit(), engine).translate_ir())[1]
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
