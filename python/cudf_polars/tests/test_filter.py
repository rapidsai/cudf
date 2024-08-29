# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize("expr", [pl.col("c"), pl.col("b") < 1, pl.lit(value=True)])
@pytest.mark.parametrize("predicate_pushdown", [False, True])
def test_filter(expr, predicate_pushdown):
    ldf = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7],
            "b": [1, 1, 1, 1, 1, 1, 1],
            "c": [True, False, False, True, True, True, None],
        }
    ).lazy()

    query = ldf.filter(expr)
    assert_gpu_result_equal(
        query, collect_kwargs={"predicate_pushdown": predicate_pushdown}
    )
