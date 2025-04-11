# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl
from polars.testing.asserts import assert_frame_equal


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
    path_polars = tmp_path / "polars.csv"
    path_cudf = tmp_path / "cudf.csv"

    df.sink_csv(
        str(path_polars),
        include_header=include_header,
        null_value=null_value,
        line_terminator=line_terminator,
        separator=separator,
    )
    df.sink_csv(
        str(path_cudf),
        engine="gpu",
        include_header=include_header,
        null_value=null_value,
        line_terminator=line_terminator,
        separator=separator,
    )

    expected = path_polars.read_text()
    result = path_cudf.read_text()
    assert result == expected


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
    path = tmp_path / "unsupported.csv"
    with pytest.raises(pl.exceptions.ComputeError):
        df.sink_csv(
            str(path), engine=pl.GPUEngine(raise_on_fail=True), **{kwarg: value}
        )


def test_sink_ndjson(df, tmp_path):
    path_polars = tmp_path / "polars.json"
    path_cudf = tmp_path / "cudf.json"

    df.sink_ndjson(str(path_polars))
    df.sink_ndjson(str(path_cudf), engine="gpu")

    expected = path_polars.read_text()
    result = path_cudf.read_text()
    # TODO: Limitation in cuDF write_json?
    result = result.replace("\\u1e85", "ẅ")
    assert result == expected


def test_sink_parquet(df, tmp_path):
    path_polars = tmp_path / "polars.pq"
    path_cudf = tmp_path / "cudf.pq"

    df.sink_parquet(str(path_polars))
    df.sink_parquet(str(path_cudf), engine="gpu")

    result = pl.read_parquet(str(path_cudf))
    expected = pl.read_parquet(str(path_polars))
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("kwarg, value", [("compression_level", 10)])
def test_sink_parquet_unsupported_kwargs(df, tmp_path, kwarg, value):
    path = tmp_path / "unsupported.pq"
    with pytest.raises(pl.exceptions.ComputeError):
        df.sink_parquet(
            str(path), engine=pl.GPUEngine(raise_on_fail=True), **{kwarg: value}
        )
