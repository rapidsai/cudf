# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import polars as pl

from cudf_polars.testing.asserts import (
    DEFAULT_CLUSTER,
    assert_gpu_result_equal,
)


def test_sort_after_sparse_join():
    engine = pl.GPUEngine(
        raise_on_fail=True,
        executor="streaming",
        executor_options={
            "cluster": DEFAULT_CLUSTER,
            "max_rows_per_partition": 4,
        },
    )

    left = pl.LazyFrame({"foo": list(range(5)), "bar": list(range(5))})
    right = pl.LazyFrame({"foo": list(range(1))})
    q = left.join(right, on="foo", how="inner").sort(by=["foo"])
    assert_gpu_result_equal(q, engine=engine)
