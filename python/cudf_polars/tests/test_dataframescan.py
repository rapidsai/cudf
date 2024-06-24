# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "subset",
    [
        None,
        ["a", "c"],
        ["b", "c", "d"],
        ["b", "d"],
        ["b", "c"],
        ["c", "e"],
        ["d", "e"],
        pl.selectors.string(),
        pl.selectors.integer(),
    ],
)
@pytest.mark.parametrize("predicate_pushdown", [False, True])
def test_scan_drop_nulls(subset, predicate_pushdown):
    df = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [None, 4, 5, None],
            "c": [6, 7, None, None],
            "d": [8, None, 9, 10],
            "e": [None, None, "A", None],
        }
    )
    # Drop nulls are pushed into filters
    q = df.drop_nulls(subset)

    assert_gpu_result_equal(
        q, collect_kwargs={"predicate_pushdown": predicate_pushdown}
    )
