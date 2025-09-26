# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal


def test_rapidmpf_scheduler_simple() -> None:
    df = pl.LazyFrame({"a": [1, 2, 3], "b": [3, 4, 5], "c": [5, 6, 7], "d": [7, 9, 8]})
    q = df.select(pl.col("a") - (pl.col("b") + pl.col("c") * 2), pl.col("d")).sort("d")

    expected = q.collect(engine="cpu")
    got_gpu = q.collect(engine=pl.GPUEngine(raise_on_fail=True))
    got_streaming = q.collect(
        engine=pl.GPUEngine(
            raise_on_fail=True,
            executor="streaming",
            executor_options={"scheduler": "rapidsmpf"},
        )
    )
    assert_frame_equal(expected, got_gpu)
    assert_frame_equal(expected, got_streaming)
