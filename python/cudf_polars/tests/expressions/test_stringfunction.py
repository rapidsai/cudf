# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from functools import partial

import pytest

import polars as pl

from cudf_polars import execute_with_cudf, translate_ir
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture
def ldf(with_nulls):
    a = [
        "AbC",
        "de",
        "FGHI",
        "j",
        "kLm",
        "nOPq",
        "",
        "RsT",
        "sada",
        "uVw",
        "h",
        "Wıth ünιcοde",  # noqa: RUF001
    ]
    if with_nulls:
        a[4] = None
        a[-3] = None
    return pl.LazyFrame({"a": a, "b": range(len(a))})


def test_supported_stringfunction_expression(ldf):
    query = ldf.select(
        pl.col("a").str.starts_with("Z"),
        pl.col("a").str.ends_with("h").alias("endswith_h"),
        pl.col("a").str.to_lowercase().alias("lower"),
        pl.col("a").str.to_uppercase().alias("upper"),
    )
    assert_gpu_result_equal(query)


def test_unsupported_stringfunction(ldf):
    q = ldf.select(pl.col("a").str.count_matches("e", literal=True))

    with pytest.raises(NotImplementedError):
        _ = translate_ir(q._ldf.visit())


def test_contains_re_non_strict_raises(ldf):
    q = ldf.select(pl.col("a").str.contains(".", strict=False))

    with pytest.raises(NotImplementedError):
        _ = translate_ir(q._ldf.visit())


def test_contains_re_non_literal_raises(ldf):
    q = ldf.select(pl.col("a").str.contains(pl.col("b"), literal=False))

    with pytest.raises(NotImplementedError):
        _ = translate_ir(q._ldf.visit())


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


def test_contains_invalid(ldf):
    query = ldf.select(pl.col("a").str.contains("["))

    with pytest.raises(pl.exceptions.ComputeError):
        query.collect()
    with pytest.raises(pl.exceptions.ComputeError):
        query.collect(post_opt_callback=partial(execute_with_cudf, raise_on_fail=True))
