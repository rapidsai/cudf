# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.mark.parametrize(
    "substr",
    [
        "A",
        "de",
        ".*",
        "^a",
        "^A",
        "[^a-z]",
        "[a-z]{3,}",
        "^[A-Z]{2,}",
        "j|u",
    ],
)
def test_contains(substr):
    ldf = pl.DataFrame(
        {"a": ["AbC", "de", "FGHI", "j", "kLm", "nOPq", None, "RsT", None, "uVw"]}
    ).lazy()

    query = ldf.select(pl.col("a").str.contains(substr))
    assert_gpu_result_equal(query)


@pytest.mark.parametrize("pat", ["["])
def test_contains_invalid(pat):
    ldf = pl.DataFrame(
        {"a": ["AbC", "de", "FGHI", "j", "kLm", "nOPq", None, "RsT", None, "uVw"]}
    ).lazy()

    query = ldf.select(pl.col("a").str.contains(pat))

    with pytest.raises(pl.exceptions.ComputeError):
        query.collect()
    with pytest.raises(pl.exceptions.ComputeError):
        query.collect(use_gpu=True)
