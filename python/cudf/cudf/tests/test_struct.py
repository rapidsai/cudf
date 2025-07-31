# Copyright (c) 2020-2025, NVIDIA CORPORATION.

import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [{}],
        [{"a": None}],
        [{"a": 1}],
        [{"a": "one"}],
        [{"a": 1}, {"a": 2}],
        [{"a": 1, "b": "one"}, {"a": 2, "b": "two"}],
        [{"b": "two", "a": None}, None, {"a": "one", "b": "two"}],
    ],
)
def test_create_struct_series(data):
    expect = pd.Series(data)
    got = cudf.Series(data)
    assert_eq(expect, got, check_dtype=False)


def test_struct_of_struct_copy():
    sr = cudf.Series([{"a": {"b": 1}}])
    assert_eq(sr, sr.copy())


def test_struct_of_struct_loc():
    df = cudf.DataFrame({"col": [{"a": {"b": 1}}]})
    expect = cudf.Series([{"a": {"b": 1}}], name="col")
    assert_eq(expect, df["col"])


@pytest.mark.parametrize(
    "key, expect", [(0, [1, 3]), (1, [2, 4]), ("a", [1, 3]), ("b", [2, 4])]
)
def test_struct_for_field(key, expect):
    sr = cudf.Series([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    expect = cudf.Series(expect)
    got = sr.struct.field(key)
    assert_eq(expect, got)


def test_series_construction_with_nulls():
    fields = [
        pa.array([1], type=pa.int64()),
        pa.array([None], type=pa.int64()),
        pa.array([3], type=pa.int64()),
    ]
    expect = pa.StructArray.from_arrays(fields, ["a", "b", "c"])
    got = cudf.Series(expect).to_arrow()

    assert expect == got
