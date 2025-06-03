# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
)


@pytest.fixture
def ldf():
    return pl.LazyFrame({"a": [{"b": "c", "d": "e"}, {"b": None, "d": "g"}]})


@pytest.mark.parametrize("name", [None, "my_count"])
def test_value_counts(ldf, name):
    # sort=True since order is non-deterministic
    query = ldf.select(pl.col("a").value_counts(sort=True, name=name))
    assert_gpu_result_equal(query)
