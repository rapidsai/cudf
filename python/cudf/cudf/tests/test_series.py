# Copyright (c) 2020, NVIDIA CORPORATION.


import pandas as pd
import pytest

import cudf
from cudf.tests.utils import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        {"a": 1, "b": 2, "c": 24, "d": 1010},
        {"a": 1},
        {1: "a", 2: "b", 24: "c", 1010: "d"},
        {1: "a"},
    ],
)
def test_series_init_dict(data):
    pandas_series = pd.Series(data)
    cudf_series = cudf.Series(data)

    assert_eq(pandas_series, cudf_series)


@pytest.mark.parametrize(
    "data",
    [
        {
            "a": [1, 2, 3],
            "b": [2, 3, 5],
            "c": [24, 12212, 22233],
            "d": [1010, 101010, 1111],
        },
        {"a": [1]},
    ],
)
def test_series_init_dict_lists(data):

    with pytest.raises(NotImplementedError):
        cudf.Series(data)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4],
        [1.0, 12.221, 12.34, 13.324, 324.3242],
        [-10, -1111, 100, 11, 133],
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        [10, 11, 12, 13],
        [0.1, 0.002, 324.2332, 0.2342],
        [-10, -1111, 100, 11, 133],
    ],
)
def test_series_append_basic(data, others):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = pd.Series(others)
    other_gs = cudf.Series(others)

    expected = psr.append(other_ps)
    actual = gsr.append(other_gs)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [
            "abc",
            "def",
            "this is a string",
            "this is another string",
            "a",
            "b",
            "c",
        ],
        ["a"],
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        [
            "abc",
            "def",
            "this is a string",
            "this is another string",
            "a",
            "b",
            "c",
        ],
        ["a"],
        ["1", "2", "3", "4", "5"],
        ["+", "-", "!", "_", "="],
    ],
)
def test_series_append_basic_str(data, others):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = pd.Series(others)
    other_gs = cudf.Series(others)

    expected = psr.append(other_ps)
    actual = gsr.append(other_gs)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        pd.Series(
            [
                "abc",
                "def",
                "this is a string",
                "this is another string",
                "a",
                "b",
                "c",
            ],
            index=[10, 20, 30, 40, 50, 60, 70],
        ),
        pd.Series(["a"], index=[2]),
    ],
)
@pytest.mark.parametrize(
    "others",
    [
        pd.Series(
            [
                "abc",
                "def",
                "this is a   string",
                "this is another string",
                "a",
                "b",
                "c",
            ],
            index=[10, 20, 30, 40, 50, 60, 70],
        ),
        pd.Series(["a"], index=[133]),
        pd.Series(["1", "2", "3", "4", "5"], index=[-10, 22, 33, 44, 49]),
        pd.Series(["+", "-", "!", "_", "="], index=[11, 22, 33, 44, 2]),
    ],
)
def test_series_append_series_with_index(data, others):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    other_ps = others
    other_gs = cudf.from_pandas(others)

    expected = psr.append(other_ps)
    actual = gsr.append(other_gs)
    assert_eq(expected, actual)
