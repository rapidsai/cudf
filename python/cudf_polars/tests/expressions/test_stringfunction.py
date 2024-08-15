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
    return pl.LazyFrame(
        {"a": a, "b": range(len(a)), "c": [str(i) for i in range(len(a))]}
    )


slice_cases = [
    (1, 3),
    (0, 3),
    (0, 0),
    (-3, 1),
    (-100, 5),
    (1, 1),
    (100, 100),
    (-3, 4),
    (-3, 3),
]


@pytest.fixture(params=slice_cases)
def slice_column_data(ldf, request):
    start, length = request.param
    if length:
        return ldf.with_columns(
            pl.lit(start).alias("start"), pl.lit(length).alias("length")
        )
    else:
        return ldf.with_columns(pl.lit(start).alias("start"))


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
    q = ldf.select(pl.col("a").str.contains(pl.col("c"), literal=False))

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


@pytest.mark.parametrize("offset", [1, -1, 0, 100, -100])
def test_slice_scalars_offset(ldf, offset):
    query = ldf.select(pl.col("a").str.slice(offset))
    assert_gpu_result_equal(query)


@pytest.mark.parametrize("offset,length", slice_cases)
def test_slice_scalars_length_and_offset(ldf, offset, length):
    query = ldf.select(pl.col("a").str.slice(offset, length))
    assert_gpu_result_equal(query)


def test_slice_column(slice_column_data):
    if "length" in slice_column_data.collect_schema():
        query = slice_column_data.select(
            pl.col("a").str.slice(pl.col("start"), pl.col("length"))
        )
    else:
        query = slice_column_data.select(pl.col("a").str.slice(pl.col("start")))
    assert_ir_translation_raises(query, NotImplementedError)


@pytest.mark.parametrize(
    "target, repl",
    [("a", "a"), ("Wı", "☺"), ("FG", ""), ("doesnotexist", "blahblah")],  # noqa: RUF001
)
@pytest.mark.parametrize("n", [0, 3, -1])
def test_replace_literal(ldf, target, repl, n):
    query = ldf.select(pl.col("a").str.replace(target, repl, literal=True, n=n))
    assert_gpu_result_equal(query)


@pytest.mark.parametrize("target, repl", [("", ""), ("a", pl.col("a"))])
def test_replace_literal_unsupported(ldf, target, repl):
    query = ldf.select(pl.col("a").str.replace(target, repl, literal=True))
    assert_ir_translation_raises(query, NotImplementedError)


def test_replace_re(ldf):
    query = ldf.select(pl.col("a").str.replace("A", "a", literal=False))
    assert_ir_translation_raises(query, NotImplementedError)


@pytest.mark.parametrize(
    "target,repl",
    [
        (["A", "de", "kLm", "awef"], "a"),
        (["A", "de", "kLm", "awef"], ""),
        (["A", "de", "kLm", "awef"], ["a", "b", "c", "d"]),
        (["A", "de", "kLm", "awef"], ["a", "b", "c", ""]),
        (
            pl.lit(pl.Series(["A", "de", "kLm", "awef"])),
            pl.lit(pl.Series(["a", "b", "c", "d"])),
        ),
    ],
)
def test_replace_many(ldf, target, repl):
    query = ldf.select(pl.col("a").str.replace_many(target, repl))

    assert_gpu_result_equal(query)


@pytest.mark.parametrize(
    "target,repl",
    [(["A", ""], ["a", "b"]), (pl.col("a").drop_nulls(), pl.col("a").drop_nulls())],
)
def test_replace_many_notimplemented(ldf, target, repl):
    query = ldf.select(pl.col("a").str.replace_many(target, repl))
    assert_ir_translation_raises(query, NotImplementedError)


def test_replace_many_ascii_case(ldf):
    query = ldf.select(
        pl.col("a").str.replace_many(["a", "b", "c"], "a", ascii_case_insensitive=True)
    )

    assert_ir_translation_raises(query, NotImplementedError)


_strip_data = [
    "AbC",
    "123abc",
    "",
    " ",
    None,
    "aAaaaAAaa",
    " ab c ",
    "abc123",
    "    ",
    "\tabc\t",
    "\nabc\n",
    "\r\nabc\r\n",
    "\t\n abc \n\t",
    "!@#$%^&*()",
    "   abc!!!   ",
    "   abc\t\n!!!   ",
    "__abc__",
    "abc\n\n",
    "123abc456",
    "abcxyzabc",
]

strip_chars = [
    "a",
    "",
    " ",
    "\t",
    "\n",
    "\r\n",
    "!",
    "@#",
    "123",
    "xyz",
    "abc",
    "__",
    " \t\n",
    "abc123",
]


@pytest.fixture
def strip_ldf():
    return pl.DataFrame({"a": _strip_data}).lazy()


@pytest.fixture(params=strip_chars)
def to_strip(request):
    return request.param


def test_strip_chars(strip_ldf, to_strip):
    q = strip_ldf.select(pl.col("a").str.strip_chars(to_strip))
    assert_gpu_result_equal(q)


def test_strip_chars_start(strip_ldf, to_strip):
    q = strip_ldf.select(pl.col("a").str.strip_chars_start(to_strip))
    assert_gpu_result_equal(q)


def test_strip_chars_end(strip_ldf, to_strip):
    q = strip_ldf.select(pl.col("a").str.strip_chars_end(to_strip))
    assert_gpu_result_equal(q)
