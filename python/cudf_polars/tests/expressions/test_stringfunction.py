# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from functools import partial

import pytest

import polars as pl

from cudf_polars import execute_with_cudf
from cudf_polars.testing.asserts import (
    assert_gpu_result_equal,
    assert_ir_translation_raises,
)


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

    assert_ir_translation_raises(q, NotImplementedError)


def test_contains_re_non_strict_raises(ldf):
    q = ldf.select(pl.col("a").str.contains(".", strict=False))

    assert_ir_translation_raises(q, NotImplementedError)


def test_contains_re_non_literal_raises(ldf):
    q = ldf.select(pl.col("a").str.contains(pl.col("b"), literal=False))

    assert_ir_translation_raises(q, NotImplementedError)


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


def test_replace_literal(ldf):
    query = ldf.select(pl.col("a").str.replace("A", "a", literal=True))
    assert_gpu_result_equal(query)


def test_replace_re(ldf):
    query = ldf.select(pl.col("a").str.replace("A", "a", literal=False))
    assert_ir_translation_raises(query, NotImplementedError)


@pytest.mark.parametrize(
    "target,repl",
    [
        (["A", "de", "kLm", "awef"], "a"),
        # TODO: this hangs on newest polars, investigate whether
        # it is a polars or cudf problem
        # (["A", "de", "kLm", "awef"], ["a", "b", "c", "d"]),
        # (pl.col("a"), pl.col("a")),
    ],
)
def test_replace_many(ldf, target, repl):
    if isinstance(target, pl.Expr):
        # libcudf cannot handle nulls in target column
        # TODO: refactor so that drop_nulls happens on the pl.col call
        # (not possible right now since the dropnull Expr function is not
        # implemented)
        ldf = ldf.drop_nulls()
    query = ldf.select(pl.col("a").str.replace_many(target, repl))

    assert_gpu_result_equal(query)


def test_replace_many_ascii_case(ldf):
    query = ldf.select(
        pl.col("a").str.replace_many(["a", "b", "c"], "a", ascii_case_insensitive=True)
    )

    assert_ir_translation_raises(query, NotImplementedError)
