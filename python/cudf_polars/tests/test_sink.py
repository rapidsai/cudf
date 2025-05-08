# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_sink_ir_translation_raises,
    assert_sink_result_equal,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_128


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
def test_sink_csv(
    request, df, tmp_path, include_header, null_value, line_terminator, separator
):
    request.applymarker(
        pytest.mark.xfail(
            condition=POLARS_VERSION_LT_128,
            reason="not supported until polars 1.28",
        )
    )
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


def test_sink_ndjson(request, df, tmp_path):
    request.applymarker(
        pytest.mark.xfail(
            condition=POLARS_VERSION_LT_128,
            reason="not supported until polars 1.28",
        )
    )
    assert_sink_result_equal(
        df,
        tmp_path / "out.ndjson",
    )


@pytest.mark.parametrize("mkdir", [True, False])
@pytest.mark.parametrize("data_page_size", [None, 256_000])
@pytest.mark.parametrize("row_group_size", [None, 1_000])
def test_sink_parquet(request, df, tmp_path, mkdir, data_page_size, row_group_size):
    request.applymarker(
        pytest.mark.xfail(
            condition=POLARS_VERSION_LT_128,
            reason="not supported until polars 1.28",
        )
    )
    assert_sink_result_equal(
        df,
        tmp_path / "out.parquet",
        write_kwargs={
            "mkdir": mkdir,
            "data_page_size": data_page_size,
            "row_group_size": row_group_size,
        },
    )


def test_sink_parquet_unsupported_kwargs(df, tmp_path):
    assert_sink_ir_translation_raises(
        df,
        tmp_path / "unsupported.parquet",
        {"compression_level": 10},
        NotImplementedError,
    )


@pytest.mark.parametrize("compression", ["gzip", "brotli", "zstd"])
def test_sink_parquet_unsupported_compression_type(df, tmp_path, compression):
    assert_sink_ir_translation_raises(
        df,
        tmp_path / "unsupported_compression.parquet",
        {"compression": compression, "compression_level": 9},
        NotImplementedError,
    )
