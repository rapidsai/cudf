# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_sink_ir_translation_raises,
    assert_sink_result_equal,
)


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "a": [1, 2, 3, None, 4, 5],
            "b": ["ẅ", "x", "y", "z", "123", "abcd"],
        }
    )


@pytest.mark.parametrize("include_header", [True, False])
@pytest.mark.parametrize("null_value", [None, "NA"])
@pytest.mark.parametrize("line_terminator", ["\n", "\n\n"])
@pytest.mark.parametrize("separator", [",", "|"])
def test_sink_csv(df, tmp_path, include_header, null_value, line_terminator, separator):
    assert_sink_result_equal(
        df,
        tmp_path / "out.csv",
        write_kwargs={
            "include_header": include_header,
            "null_value": null_value,
            "line_terminator": line_terminator,
            "separator": separator,
        },
    )


@pytest.mark.parametrize(
    "kwarg, value",
    [
        ("include_bom", True),
        ("date_format", "%Y-%m-%d"),
        ("time_format", "%H:%M:%S"),
        ("datetime_format", "%Y-%m-%dT%H:%M:%S"),
        ("float_scientific", True),
        ("float_precision", 10),
        ("quote_style", "non_numeric"),
        ("quote_char", "`"),
    ],
)
def test_sink_csv_unsupported_kwargs(df, tmp_path, kwarg, value):
    assert_sink_ir_translation_raises(
        df,
        tmp_path / "unsupported.csv",
        {kwarg: value},
        NotImplementedError,
    )


def test_sink_ndjson(df, tmp_path):
    assert_sink_result_equal(
        df,
        tmp_path / "out.ndjson",
    )


def test_sink_parquet(df, tmp_path):
    assert_sink_result_equal(
        df,
        tmp_path / "out.parquet",
    )


def test_sink_parquet_unsupported_kwargs(df, tmp_path):
    assert_sink_ir_translation_raises(
        df,
        tmp_path / "unsupported.parquet",
        {"compression_level": 10},
        NotImplementedError,
    )
