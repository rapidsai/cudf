# Copyright (c) 2025, NVIDIA CORPORATION.

import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [1],
        [1, 2, 3],
        [[1, 2, 3], [4, 5, 6]],
        [pd.NA],
        [1, pd.NA, 3],
        [[1, pd.NA, 3], [pd.NA, 5, 6]],
        [[1.1, pd.NA, 3.3], [4.4, 5.5, pd.NA]],
        [["a", pd.NA, "c"], ["d", "e", pd.NA]],
        [["a", "b", "c"], ["d", "e", "f"]],
    ],
)
def test_list_getitem(data):
    list_sr = cudf.Series([data])
    assert list_sr[0] == data


@pytest.mark.parametrize("nesting_level", [1, 3])
def test_list_scalar_device_construction_null(nesting_level):
    data = [[]]
    for i in range(nesting_level - 1):
        data = [data]

    arrow_type = pa.infer_type(data)
    arrow_arr = pa.array([None], type=arrow_type)

    res = cudf.Series(arrow_arr)[0]
    assert res is cudf.NA


@pytest.mark.parametrize(
    "data, idx",
    [
        (
            [[{"f2": {"a": 100}, "f1": "a"}, {"f1": "sf12", "f2": pd.NA}]],
            0,
        ),
        (
            [
                [
                    {"f2": {"a": 100, "c": 90, "f2": 10}, "f1": "a"},
                    {"f1": "sf12", "f2": pd.NA},
                ]
            ],
            0,
        ),
        (
            [[[[1, 2]], [[2], [3]]], [[[2]]], [[[3]]]],
            0,
        ),
        ([[[[1, 2]], [[2], [3]]], [[[2]]], [[[3]]]], 2),
        ([[[{"a": 1, "b": 2, "c": 10}]]], 0),
    ],
)
def test_nested_list_extract_host_scalars(data, idx):
    series = cudf.Series(data)

    assert series[idx] == data[idx]


@pytest.mark.parametrize(
    "data, idx, expected",
    [
        (
            [{"f2": {"a": "sf21"}, "f1": "a"}, {"f1": "sf12", "f2": None}],
            0,
            {"f1": "a", "f2": {"a": "sf21"}},
        ),
        (
            [
                {"f2": {"a": "sf21"}},
                {"f1": "sf12", "f2": None},
            ],
            0,
            {"f1": cudf.NA, "f2": {"a": "sf21"}},
        ),
        (
            [{"a": "123"}, {"a": "sf12", "b": {"a": {"b": "c"}}}],
            1,
            {"a": "sf12", "b": {"a": {"b": "c"}}},
        ),
    ],
)
def test_nested_struct_extract_host_scalars(data, idx, expected):
    series = cudf.Series(data)

    def _nested_na_replace(struct_scalar):
        """
        Replace `cudf.NA` with `None` in the dict
        """
        for key, value in struct_scalar.items():
            if value is cudf.NA:
                struct_scalar[key] = None
        return struct_scalar

    assert _nested_na_replace(series[idx]) == _nested_na_replace(expected)


def test_nested_struct_from_pandas_empty():
    # tests constructing nested structs columns that would result in
    # libcudf EMPTY type child columns inheriting their parent's null
    # mask. See GH PR: #10761
    pdf = pd.Series([[{"c": {"x": None}}], [{"c": None}]])
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf, gdf)


def test_struct_int_values():
    series = cudf.Series(
        [{"a": 1, "b": 2}, {"a": 10, "b": None}, {"a": 5, "b": 6}]
    )
    actual_series = series.to_pandas()

    assert isinstance(actual_series[0]["b"], int)
    assert isinstance(actual_series[1]["b"], type(None))
    assert isinstance(actual_series[2]["b"], int)


def test_struct_slice_nested_struct():
    data = [
        {"a": {"b": 42, "c": "abc"}},
        {"a": {"b": 42, "c": "hello world"}},
    ]

    got = cudf.Series(data)[0:1]
    expect = cudf.Series(data[0:1])
    assert got.to_arrow() == expect.to_arrow()


@pytest.mark.parametrize(
    "series, slce",
    [
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": 1},
                {},
                None,
            ],
            slice(1, None),
        ),
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": 1},
                {},
                None,
                {"d": ["Hello", "rapids"]},
                None,
                cudf.NA,
            ],
            slice(1, 5),
        ),
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": 1},
                {},
                None,
                {"c": 5},
                None,
                cudf.NA,
            ],
            slice(None, 4),
        ),
        ([{"a": {"b": 42, "c": -1}}, {"a": {"b": 0, "c": None}}], slice(0, 1)),
    ],
)
def test_struct_slice(series, slce):
    got = cudf.Series(series)[slce]
    expected = cudf.Series(series[slce])
    assert got.to_arrow() == expected.to_arrow()


@pytest.mark.parametrize(
    "series, expected",
    [
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": 1},
                {},
            ],
            {"a": "Hello world", "b": [], "c": cudf.NA},
        ),
        ([{}], {}),
        (
            [{"b": True}, {"a": 1, "c": [1, 2, 3], "d": "1", "b": False}],
            {"a": cudf.NA, "c": cudf.NA, "d": cudf.NA, "b": True},
        ),
    ],
)
def test_struct_getitem(series, expected):
    sr = cudf.Series(series)
    assert sr[0] == expected


def test_datetime_getitem_na():
    s = cudf.Series([1, 2, None, 3], dtype="datetime64[ns]")
    assert s[2] is cudf.NaT


def test_timedelta_getitem_na():
    s = cudf.Series([1, 2, None, 3], dtype="timedelta64[ns]")
    assert s[2] is cudf.NaT
