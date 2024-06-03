# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.xfail(reason="Need handling of null scalars that are cast")
def test_union():
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    ).lazy()
    ldf2 = ldf.select((pl.col("a") + pl.col("b")).alias("c"), pl.col("a"))
    query = pl.concat([ldf, ldf2], how="diagonal")
    # Plan for this produces a `None`.astype(Int64) which we don't
    # handle correctly right now
    assert_gpu_result_equal(query)


def test_concat_vertical():
    ldf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
        }
    )
    ldf2 = ldf.select(pl.col("a"), pl.col("b") * 2 + pl.col("a"))
    q = pl.concat([ldf, ldf2], how="vertical")

    assert_gpu_result_equal(q)
