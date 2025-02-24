# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_merge_sorted():
    df0 = pl.LazyFrame(
        {"name": ["steve", "elise", "bob"], "age": [42, 44, 18]}
    ).sort("age")
    df1 = pl.LazyFrame(
        {"name": ["anna", "megan", "steve", "thomas"], "age": [21, 33, 42, 20]}
    ).sort("age")
    q = df0.merge_sorted(df1, key="age")
    assert_gpu_result_equal(q)

