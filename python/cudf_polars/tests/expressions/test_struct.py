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
@pytest.mark.parametrize("normalize", [True, False])
def test_value_counts(ldf, name, normalize):
    # sort=True since order is non-deterministic
    query = ldf.select(
        pl.col("a").value_counts(sort=True, name=name, normalize=normalize)
    )
    assert_gpu_result_equal(query)


def test_struct(ldf):
    query = ldf.select(pl.struct(pl.all()))
    assert_gpu_result_equal(query)


def test_nested_struct():
    ldf = pl.LazyFrame({"a": [{"x": {"i": 0, "j": 0}, "y": {"i": 0, "k": 1}}]})
    query = ldf.select(pl.struct(pl.all()))
    assert_gpu_result_equal(query)
