# Copyright (c) 2020-2024, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core.dtypes import StructDtype
from cudf.testing import assert_eq
from cudf.testing._utils import DATETIME_TYPES, TIMEDELTA_TYPES


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


@pytest.mark.parametrize(
    "fields",
    [
        {"a": np.dtype(np.int64)},
        {"a": np.dtype(np.int64), "b": None},
        {
            "a": cudf.ListDtype(np.dtype(np.int64)),
            "b": cudf.Decimal64Dtype(1, 0),
        },
        {
            "a": cudf.ListDtype(cudf.StructDtype({"b": np.dtype(np.int64)})),
            "b": cudf.ListDtype(cudf.ListDtype(np.dtype(np.int64))),
        },
    ],
)
def test_serialize_struct_dtype(fields):
    dtype = cudf.StructDtype(fields)
    recreated = dtype.__class__.deserialize(*dtype.serialize())
    assert recreated == dtype


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


@pytest.mark.parametrize(
    "data, item",
    [
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": cudf.NA},
                {"a": "abcde", "b": [4, 5, 6], "c": 9},
            ],
            {"a": "Hello world", "b": [], "c": cudf.NA},
        ),
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": cudf.NA},
                {"a": "abcde", "b": [4, 5, 6], "c": 9},
            ],
            {},
        ),
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": cudf.NA},
                {"a": "abcde", "b": [4, 5, 6], "c": 9},
            ],
            cudf.NA,
        ),
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": cudf.NA},
                {"a": "abcde", "b": [4, 5, 6], "c": 9},
            ],
            {"a": "Second element", "b": [1, 2], "c": 1000},
        ),
    ],
)
def test_struct_setitem(data, item):
    sr = cudf.Series(data)
    sr[1] = item
    data[1] = item
    expected = cudf.Series(data)
    assert sr.to_arrow() == expected.to_arrow()


@pytest.mark.parametrize(
    "data",
    [
        {"a": 1, "b": "rapids", "c": [1, 2, 3, 4]},
        {"a": "Hello"},
    ],
)
def test_struct_scalar_host_construction(data):
    slr = cudf.Scalar(data)
    assert slr.value == data
    assert list(slr.device_value.value.values()) == list(data.values())


@pytest.mark.parametrize(
    ("data", "dtype"),
    [
        (
            {"a": 1, "b": "rapids", "c": [1, 2, 3, 4], "d": cudf.NA},
            cudf.StructDtype(
                {
                    "a": np.dtype(np.int64),
                    "b": np.dtype(np.str_),
                    "c": cudf.ListDtype(np.dtype(np.int64)),
                    "d": np.dtype(np.int64),
                }
            ),
        ),
        (
            {"b": [], "c": [1, 2, 3]},
            cudf.StructDtype(
                {
                    "b": cudf.ListDtype(np.dtype(np.int64)),
                    "c": cudf.ListDtype(np.dtype(np.int64)),
                }
            ),
        ),
    ],
)
def test_struct_scalar_host_construction_no_dtype_inference(data, dtype):
    # cudf cannot infer the dtype of the scalar when it contains only nulls or
    # is empty.
    slr = cudf.Scalar(data, dtype=dtype)
    assert slr.value == data
    assert list(slr.device_value.value.values()) == list(data.values())


def test_struct_scalar_null():
    slr = cudf.Scalar(cudf.NA, dtype=StructDtype)
    assert slr.device_value.value is cudf.NA


def test_struct_explode():
    s = cudf.Series([], dtype=cudf.StructDtype({}))
    expect = cudf.DataFrame({})
    assert_eq(expect, s.struct.explode())

    s = cudf.Series(
        [
            {"a": 1, "b": "x"},
            {"a": 2, "b": "y"},
            {"a": 3, "b": "z"},
            {"a": 4, "b": "a"},
        ]
    )
    expect = cudf.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "z", "a"]})
    got = s.struct.explode()
    assert_eq(expect, got)

    # check that a copy was made:
    got["a"][0] = 5
    assert_eq(s.struct.explode(), expect)


def test_dataframe_to_struct():
    df = cudf.DataFrame()
    expect = cudf.Series(dtype=cudf.StructDtype({}))
    got = df.to_struct()
    assert_eq(expect, got)

    df = cudf.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    expect = cudf.Series(
        [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]
    )
    got = df.to_struct()
    assert_eq(expect, got)

    # check that a copy was made:
    df["a"][0] = 5
    assert_eq(got, expect)

    # check that a non-string (but convertible to string) named column can be
    # converted to struct
    df = cudf.DataFrame([[1, 2], [3, 4]], columns=[(1, "b"), 0])
    expect = cudf.Series([{"(1, 'b')": 1, "0": 2}, {"(1, 'b')": 3, "0": 4}])
    with pytest.warns(UserWarning, match="will be casted"):
        got = df.to_struct()
    assert_eq(got, expect)


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


def test_struct_slice_nested_struct():
    data = [
        {"a": {"b": 42, "c": "abc"}},
        {"a": {"b": 42, "c": "hello world"}},
    ]

    got = cudf.Series(data)[0:1]
    expect = cudf.Series(data[0:1])
    assert got.to_arrow() == expect.to_arrow()


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
def test_struct_field_errors(data):
    got = cudf.Series(data)

    with pytest.raises(KeyError):
        got.struct.field("notWithinFields")

    with pytest.raises(IndexError):
        got.struct.field(100)


@pytest.mark.parametrize("dtype", DATETIME_TYPES + TIMEDELTA_TYPES)
def test_struct_with_datetime_and_timedelta(dtype):
    df = cudf.DataFrame(
        {
            "a": [12, 232, 2334],
            "datetime": cudf.Series([23432, 3432423, 324324], dtype=dtype),
        }
    )
    series = df.to_struct()
    a_array = np.array([12, 232, 2334])
    datetime_array = np.array([23432, 3432423, 324324]).astype(dtype)

    actual = series.to_pandas()
    values_list = []
    for i, val in enumerate(a_array):
        values_list.append({"a": val, "datetime": datetime_array[i]})

    expected = pd.Series(values_list)
    assert_eq(expected, actual)


def test_struct_int_values():
    series = cudf.Series(
        [{"a": 1, "b": 2}, {"a": 10, "b": None}, {"a": 5, "b": 6}]
    )
    actual_series = series.to_pandas()

    assert isinstance(actual_series[0]["b"], int)
    assert isinstance(actual_series[1]["b"], type(None))
    assert isinstance(actual_series[2]["b"], int)


def test_nested_struct_from_pandas_empty():
    # tests constructing nested structs columns that would result in
    # libcudf EMPTY type child columns inheriting their parent's null
    # mask. See GH PR: #10761
    pdf = pd.Series([[{"c": {"x": None}}], [{"c": None}]])
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf, gdf)


def _nested_na_replace(struct_scalar):
    """
    Replace `cudf.NA` with `None` in the dict
    """
    for key, value in struct_scalar.items():
        if value is cudf.NA:
            struct_scalar[key] = None
    return struct_scalar


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

    assert _nested_na_replace(series[idx]) == _nested_na_replace(expected)


def test_struct_memory_usage():
    s = cudf.Series([{"a": 1, "b": 10}, {"a": 2, "b": 20}, {"a": 3, "b": 30}])
    df = s.struct.explode()

    assert_eq(s.memory_usage(), df.memory_usage().sum())


def test_struct_with_null_memory_usage():
    df = cudf.DataFrame(
        {
            "a": cudf.Series([1, 2, -1, -1, 3], dtype="int64"),
            "b": cudf.Series([10, 20, -1, -1, 30], dtype="int64"),
        }
    )
    s = df.to_struct()
    assert s.memory_usage() == 80

    s[2:4] = None
    assert s.memory_usage() == 272


@pytest.mark.parametrize(
    "indices",
    [slice(0, 3), slice(1, 4), slice(None, None, 2), slice(1, None, 2)],
    ids=[":3", "1:4", "0::2", "1::2"],
)
@pytest.mark.parametrize(
    "values",
    [[None, {}, {}, None], [{}, {}, {}, {}]],
    ids=["nulls", "no_nulls"],
)
def test_struct_empty_children_slice(indices, values):
    s = cudf.Series(values)
    actual = s.iloc[indices]
    expect = cudf.Series(values[indices], index=range(len(values))[indices])
    assert_eq(actual, expect)


def test_struct_iterate_error():
    s = cudf.Series(
        [{"f2": {"a": "sf21"}, "f1": "a"}, {"f1": "sf12", "f2": None}]
    )
    with pytest.raises(TypeError):
        iter(s.struct)
