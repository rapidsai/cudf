# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "series, value",
    [
        (pl.Series("a", [1, 2, 2, 2, 4, 5]), 2),
        (pl.Series("a", [1, 2, 3]), 99),
        (pl.Series("a", [5, 4, 3, 2, 1]), 3),
        (pl.Series("a", [1, None, 3]), None),
        (pl.Series("a", [1, None, 3]), 3),
        (pl.Series("a", [1.5, float("nan"), 3.0]), 3.0),
        (pl.Series("a", ["a", "bb", "c"]), "bb"),
        (pl.Series("a", ["a", "bb", "c"]), "zz"),
        (pl.Series("a", [], dtype=pl.Int64), 1),
    ],
)
def test_index_of(
    engine: pl.GPUEngine, series: pl.Series, value: int | float | str | None
) -> None:
    lf = pl.LazyFrame({"a": series})
    q = lf.select(pl.col("a").index_of(value))
    assert_gpu_result_equal(q, engine=engine)


def test_index_of_expression_value(engine: pl.GPUEngine) -> None:
    lf = pl.LazyFrame(
        {
            "a": [1, None, 3],
            "needle": pl.Series([None, None, None], dtype=pl.Int64),
        }
    )
    q = lf.select(pl.col("a").index_of(pl.col("needle").first()))
    assert_gpu_result_equal(q, engine=engine)
