# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars import translate_ir
from cudf_polars.testing.asserts import assert_gpu_result_equal


def test_supported_stringfunction_expression():
    ldf = pl.LazyFrame(
        {
            "a": ["a", "b", "cdefg", "h", "Wıth ünιcοde"],  # noqa: RUF001
            "b": [0, 3, 1, -1, None],
        }
    )

    query = ldf.select(
        pl.col("a").str.starts_with("Z"),
        pl.col("a").str.ends_with("h").alias("endswith_h"),
        pl.col("a").str.to_lowercase().alias("lower"),
        pl.col("a").str.to_uppercase().alias("upper"),
    )
    assert_gpu_result_equal(query)


def test_unsupported_stringfunction():
    ldf = pl.LazyFrame(
        {
            "a": ["a", "b", "cdefg", "h", "Wıth ünιcοde"],  # noqa: RUF001
            "b": [0, 3, 1, -1, None],
        }
    )

    q = ldf.select(pl.col("a").str.count_matches("e", literal=True))

    with pytest.raises(NotImplementedError):
        _ = translate_ir(q._ldf.visit())
