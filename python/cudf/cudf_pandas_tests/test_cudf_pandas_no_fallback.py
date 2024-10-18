# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cudf.pandas import LOADED

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

import numpy as np
import pandas as pd


@pytest.fixture(autouse=True)
def fail_on_fallback(monkeypatch):
    monkeypatch.setenv("CUDF_PANDAS_FAIL_ON_FALLBACK", "True")


@pytest.fixture
def dataframe():
    df = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 3],
            "b": [1, 2, 3, 4, 5],
            "c": [1.2, 1.3, 1.5, 1.7, 1.11],
        }
    )
    return df


@pytest.fixture
def series(dataframe):
    return dataframe["a"]


@pytest.fixture
def array(series):
    return series.values


@pytest.mark.parametrize(
    "op",
    [
        "sum",
        "min",
        "max",
        "mean",
        "std",
        "var",
        "prod",
        "median",
    ],
)
def test_no_fallback_in_reduction_ops(series, op):
    s = series
    getattr(s, op)()


def test_groupby(dataframe):
    df = dataframe
    df.groupby("a", sort=True).max()


def test_no_fallback_in_binops(dataframe):
    df = dataframe
    df + df
    df - df
    df * df
    df**df
    df[["a", "b"]] & df[["a", "b"]]
    df <= df


def test_no_fallback_in_groupby_rolling_sum(dataframe):
    df = dataframe
    df.groupby("a").rolling(2).sum()


def test_no_fallback_in_concat(dataframe):
    df = dataframe
    pd.concat([df, df])


def test_no_fallback_in_get_shape(dataframe):
    df = dataframe
    df.shape


def test_no_fallback_in_array_ufunc_op(array):
    np.add(array, array)


def test_no_fallback_in_merge(dataframe):
    df = dataframe
    pd.merge(df * df, df + df, how="inner")
    pd.merge(df * df, df + df, how="outer")
    pd.merge(df * df, df + df, how="left")
    pd.merge(df * df, df + df, how="right")
