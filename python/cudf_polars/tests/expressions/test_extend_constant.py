# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "series, value",
    [
        (pl.Series([1, 2, 3]), 0),
        (pl.Series([1, 2, None]), None),
        (pl.Series([1.5, float("nan"), -3.0]), 2.5),
        (pl.Series([1.5, 2.5, -3.0]), None),
        (pl.Series(["a", "bb", None]), "zz"),
        (pl.Series(["a", "bb", "c"]), None),
        (pl.Series([], dtype=pl.Int64), 0),
    ],
)
@pytest.mark.parametrize("n", [0, 1, 3])
def test_extend_constant(
    engine: pl.GPUEngine, series: pl.Series, value: int | float | str | None, n: int
) -> None:
    lf = pl.LazyFrame({"a": series})
    q = lf.select(pl.col("a").extend_constant(value, n))
    assert_gpu_result_equal(q, engine=engine)
