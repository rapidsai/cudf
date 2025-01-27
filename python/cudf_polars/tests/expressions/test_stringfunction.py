# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from functools import partial

import pytest

import polars as pl

from cudf_polars import execute_with_cudf
from cudf_polars.testing.asserts import (
    assert_collect_raises,
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


@pytest.fixture(params=[pl.Int8, pl.Int16, pl.Int32, pl.Int64])
def integer_type(request):
    return request.param


@pytest.fixture(params=[pl.Float32, pl.Float64])
def floating_type(request):
    return request.param


@pytest.fixture(params=[pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.Float32, pl.Float64])
def numeric_type(request):
    return request.param


@pytest.fixture
def str_to_integer_data(with_nulls):
    a = ["1", "2", "3", "4", "5", "6"]
    if with_nulls:
        a[4] = None
    return pl.LazyFrame({"a": a})


@pytest.fixture
def str_to_float_data(with_nulls):
    a = [
        "1.1",
        "2.2",
        "3.3",
        "4.4",
        "5.5",
        "6.6",
        "inf",
        "+inf",
        "-inf",
        "Inf",
        "-Inf",
        "nan",
        "-1.234",
        "2e2",
    ]
    if with_nulls:
        a[4] = None
    return pl.LazyFrame({"a": a})


@pytest.fixture
def str_from_integer_data(with_nulls, integer_type):
    a = [1, 2, 3, 4, 5, 6]
    if with_nulls:
        a[4] = None
    return pl.LazyFrame({"a": pl.Series(a, dtype=integer_type)})


@pytest.fixture
def str_from_float_data(with_nulls, floating_type):
    a = [
        1.1,
        2.2,
        3.3,
        4.4,
        5.5,
        6.6,
        float("inf"),
        float("+inf"),
        float("-inf"),
        float("nan"),
    ]
    if with_nulls:
        a[4] = None
    return pl.LazyFrame({"a": pl.Series(a, dtype=floating_type)})


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


@pytest.fixture
def to_datetime_data():
    return pl.LazyFrame(
        {
            "a": [
                "2021-01-01",
                "2021-01-02",
                "abcd",
            ]
        }
    )


@pytest.mark.parametrize("cache", [True, False], ids=lambda cache: f"{cache=}")
@pytest.mark.parametrize("strict", [True, False], ids=lambda strict: f"{strict=}")
@pytest.mark.parametrize("exact", [True, False], ids=lambda exact: f"{exact=}")
@pytest.mark.parametrize("format", ["%Y-%m-%d", None], ids=lambda format: f"{format=}")
def test_to_datetime(to_datetime_data, cache, strict, format, exact):
    query = to_datetime_data.select(
        pl.col("a").str.strptime(
            pl.Datetime("ns"), format=format, cache=cache, strict=strict, exact=exact
        )
    )
    if cache or format is None or not exact:
        assert_ir_translation_raises(query, NotImplementedError)
    elif strict:
        assert_collect_raises(
            query,
            polars_except=pl.exceptions.InvalidOperationError,
            cudf_except=pl.exceptions.ComputeError,
        )
    else:
        assert_gpu_result_equal(query)


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
    None,
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


def test_strip_chars_column(strip_ldf):
    q = strip_ldf.select(pl.col("a").str.strip_chars(pl.col("a")))
    assert_ir_translation_raises(q, NotImplementedError)


def test_invalid_regex_raises():
    df = pl.LazyFrame({"a": ["abc"]})

    q = df.select(pl.col("a").str.contains(r"ab)", strict=True))

    assert_collect_raises(
        q,
        polars_except=pl.exceptions.ComputeError,
        cudf_except=pl.exceptions.ComputeError,
    )


@pytest.mark.parametrize("pattern", ["a{1000}", "a(?i:B)"])
def test_unsupported_regex_raises(pattern):
    df = pl.LazyFrame({"a": ["abc"]})

    q = df.select(pl.col("a").str.contains(pattern, strict=True))
    assert_ir_translation_raises(q, NotImplementedError)


def test_string_to_integer(str_to_integer_data, integer_type):
    query = str_to_integer_data.select(pl.col("a").cast(integer_type))
    assert_gpu_result_equal(query)


def test_string_from_integer(str_from_integer_data):
    query = str_from_integer_data.select(pl.col("a").cast(pl.String))
    assert_gpu_result_equal(query)


def test_string_to_float(str_to_float_data, floating_type):
    query = str_to_float_data.select(pl.col("a").cast(floating_type))
    assert_gpu_result_equal(query)


def test_string_from_float(request, str_from_float_data):
    if str_from_float_data.collect_schema()["a"] == pl.Float32:
        # libcudf will return a string representing the precision out to
        # a certain number of hardcoded decimal places. This results in
        # the fractional part being thrown away which causes discrepancies
        # for certain numbers. For instance, the float32 representation of
        # 1.1 is 1.100000023841858. When cast to a string, this will become
        # 1.100000024. But the float64 representation of 1.1 is
        # 1.1000000000000000888 which will result in libcudf truncating the
        # final value to 1.1.
        request.applymarker(pytest.mark.xfail(reason="libcudf truncation"))
    query = str_from_float_data.select(pl.col("a").cast(pl.String))

    # libcudf reads float('inf') -> "inf"
    # but polars reads float('inf') -> "Inf"
    query = query.select(pl.col("a").str.to_lowercase())
    assert_gpu_result_equal(query)


def test_string_to_numeric_invalid(numeric_type):
    df = pl.LazyFrame({"a": ["a", "b", "c"]})
    q = df.select(pl.col("a").cast(numeric_type))
    assert_collect_raises(
        q,
        polars_except=pl.exceptions.InvalidOperationError,
        cudf_except=pl.exceptions.ComputeError,
    )
