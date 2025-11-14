# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import re

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2.5, 3], "b": [3, 4.5, 5], "c": [2.0, 3.0, 4.0]},
        {"a": [1, 2.2, 3], "b": [2.0, 3.0, 4.0], "c": [5.0, 6.0, 4.0]},
    ],
)
@pytest.mark.parametrize(
    "aggs",
    [
        ["min", "sum", "max"],
        ("min", "sum", "max"),
        {"min", "sum", "max"},
        "sum",
        {"a": "sum", "b": "min", "c": "max"},
        {"a": ["sum"], "b": ["min"], "c": ["max"]},
        {"a": ("sum"), "b": ("min"), "c": ("max")},
        {"a": {"sum"}, "b": {"min"}, "c": {"max"}},
        {"a": ["sum", "min"], "b": ["sum", "max"], "c": ["min", "max"]},
        {"a": ("sum", "min"), "b": ("sum", "max"), "c": ("min", "max")},
        {"a": {"sum", "min"}, "b": {"sum", "max"}, "c": {"min", "max"}},
    ],
)
def test_agg_for_dataframes(data, aggs):
    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)

    expect = pdf.agg(aggs).sort_index()
    got = gdf.agg(aggs).sort_index()

    assert_eq(expect, got, check_dtype=True)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3], "b": [3.0, 4.0, 5.0], "c": [True, True, False]},
        {"a": [1, 2, 3], "b": [True, True, False], "c": [False, True, False]},
    ],
)
@pytest.mark.parametrize(
    "aggs",
    [
        ["min", "sum", "max"],
        "sum",
        {"a": "sum", "b": "min", "c": "max"},
    ],
)
def test_agg_for_dataframes_error(data, aggs):
    gdf = cudf.DataFrame(data)

    with pytest.raises(TypeError):
        gdf.agg(aggs)


def test_agg_for_unsupported_function():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]})

    with pytest.raises(NotImplementedError):
        gdf.agg({"a": np.sum, "b": np.min, "c": np.max})


def test_agg_for_dataframe_with_invalid_function():
    aggs = "asdf"
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]})

    with pytest.raises(
        AttributeError,
        match=f"{aggs} is not a valid function for 'DataFrame' object",
    ):
        gdf.agg(aggs)


def test_agg_for_series_with_invalid_function():
    aggs = {"a": "asdf"}
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [3.0, 4.0, 5.0]})

    with pytest.raises(
        AttributeError,
        match=f"{aggs['a']} is not a valid function for 'Series' object",
    ):
        gdf.agg(aggs)


@pytest.mark.parametrize(
    "aggs",
    [
        "sum",
        ["min", "sum", "max"],
        {"a": {"sum", "min"}, "b": {"sum", "max"}, "c": {"min", "max"}},
    ],
)
def test_agg_for_dataframe_with_string_columns(aggs):
    gdf = cudf.DataFrame(
        {"a": ["m", "n", "o"], "b": ["t", "u", "v"], "c": ["x", "y", "z"]},
        index=["a", "b", "c"],
    )

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "DataFrame.agg() is not supported for "
            "frames containing string columns"
        ),
    ):
        gdf.agg(aggs)
