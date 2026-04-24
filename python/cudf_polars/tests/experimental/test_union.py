# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_union_shared_fanout_no_deadlock(streaming_engine):
    # union actor can deadlock when input branches share a fanout.
    # See https://github.com/rapidsai/cudf/issues/21750
    n = 100
    df = pl.LazyFrame({"key": list(range(50)) * (n // 50), "val": list(range(n))})
    gb = df.group_by("key").agg(pl.col("val").sum())
    project = df.select("key", "val")
    q = pl.concat([gb, project])
    assert_gpu_result_equal(q, engine=streaming_engine, check_row_order=False)
