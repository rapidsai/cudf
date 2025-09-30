# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.mark.parametrize("rows_per_partition", [1, 10, 20])
def test_rapidmpf_scheduler_simple(rows_per_partition: int) -> None:
    df = pl.LazyFrame(
        {
            "a": list(range(0, 20)),
            "b": list(range(20, 40)),
            "c": list(range(40, 60)),
            "d": list(range(60, 80)),
        }
    )
    q = df.select(pl.col("a") - (pl.col("b") + pl.col("c") * 2), pl.col("d")).sort("d")

    expected = q.collect(engine="cpu")
    got_gpu = q.collect(engine=pl.GPUEngine(raise_on_fail=True))
    got_streaming = q.collect(
        engine=pl.GPUEngine(
            raise_on_fail=True,
            executor="streaming",
            executor_options={
                "scheduler": "rapidsmpf",
                "max_rows_per_partition": rows_per_partition,
            },
        )
    )
    assert_frame_equal(expected, got_gpu)
    assert_frame_equal(expected, got_streaming)
