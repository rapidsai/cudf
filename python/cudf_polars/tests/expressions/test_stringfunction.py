# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from functools import partial

import pytest

import polars as pl

from cudf_polars import execute_with_cudf, translate_ir
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


def test_contains_re_non_strict_raises():
    df = pl.LazyFrame({"a": ["a"]})

    q = df.select(pl.col("a").str.contains(".", strict=False))

    with pytest.raises(NotImplementedError):
        _ = translate_ir(q._ldf.visit())


@pytest.fixture
def ldf():
    return pl.DataFrame(
        {"a": ["AbC", "de", "FGHI", "j", "kLm", "nOPq", None, "RsT", None, "uVw"]}
    ).lazy()


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
def test_contains_regex(ldf, substr):
    query = ldf.select(pl.col("a").str.contains(substr))
    assert_gpu_result_equal(query)


@pytest.mark.parametrize(
    "literal", ["A", "de", "FGHI", "j", "kLm", "nOPq", "RsT", "uVw"]
)
def test_contains_literal(ldf, literal):
    query = ldf.select(pl.col("a").str.contains(pl.lit(literal), literal=True))
    assert_gpu_result_equal(query)


def test_contains_column(ldf):
    query = ldf.select(pl.col("a").str.contains(pl.col("a"), literal=True))
    assert_gpu_result_equal(query)


@pytest.mark.parametrize("pat", ["["])
def test_contains_invalid(ldf, pat):
    query = ldf.select(pl.col("a").str.contains(pat))

    with pytest.raises(pl.exceptions.ComputeError):
        query.collect()
    with pytest.raises(pl.exceptions.ComputeError):
        query.collect(post_opt_callback=partial(execute_with_cudf, raise_on_fail=True))
