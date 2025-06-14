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


def test_struct(ldf):
    query = ldf.select(pl.struct(pl.all()))
    assert_gpu_result_equal(query)
