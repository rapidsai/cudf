# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from functools import partial

import pytest

import polars as pl

from cudf_polars.callback import execute_with_cudf
from cudf_polars.dsl.translate import translate_ir
from cudf_polars.testing.asserts import assert_gpu_result_equal


@pytest.fixture
def ldf():
    return pl.DataFrame(
        {"a": ["AbC", "de", "FGHI", "ja", "kLm", "nAOPq", None, "RsT", None, "uVw"]}
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


def test_replace_literal(ldf):
    query = ldf.select(pl.col("a").str.replace("A", "a", literal=True))
    assert_gpu_result_equal(query)


def test_replace_re(ldf):
    query = ldf.select(pl.col("a").str.replace("A", "a", literal=False))
    with pytest.raises(NotImplementedError):
        _ = translate_ir(query._ldf.visit())


@pytest.mark.parametrize(
    "target,repl",
    [
        (["A", "de", "kLm", "awef"], "a"),
        (["A", "de", "kLm", "awef"], ["a", "b", "c", "d"]),
        (pl.col("a"), pl.col("a")),
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

    with pytest.raises(NotImplementedError):
        _ = translate_ir(query._ldf.visit())
