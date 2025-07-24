# Copyright (c) 2023-2025, NVIDIA CORPORATION.
import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core.column.column import as_column
from cudf.testing import assert_eq


def test_construct_int_series_with_nulls_compat_mode():
    # in compatibility mode, constructing a Series
    # with nulls should result in a floating Series:
    with cudf.option_context("mode.pandas_compatible", True):
        s = cudf.Series([1, 2, None])
    assert s.dtype == np.dtype("float64")


@pytest.mark.parametrize(
    "data",
    [
        {"a": 1, "b": 2, "c": 24, "d": 1010},
        {"a": 1},
        {1: "a", 2: "b", 24: "c", 1010: "d"},
        {1: "a"},
        {"a": [1]},
    ],
)
def test_series_init_dict(data):
    pandas_series = pd.Series(data)
    cudf_series = cudf.Series(data)

    assert_eq(pandas_series, cudf_series)


def test_series_unitness_np_datetimelike_units():
    data = np.array([np.timedelta64(1)])
    with pytest.raises(TypeError):
        cudf.Series(data)
    with pytest.raises(TypeError):
        pd.Series(data)


def test_list_category_like_maintains_dtype():
    dtype = cudf.CategoricalDtype(categories=[1, 2, 3, 4], ordered=True)
    data = [1, 2, 3]
    result = cudf.Series._from_column(as_column(data, dtype=dtype))
    expected = pd.Series(data, dtype=dtype.to_pandas())
    assert_eq(result, expected)


def test_list_interval_like_maintains_dtype():
    dtype = cudf.IntervalDtype(subtype=np.int8)
    data = [pd.Interval(1, 2)]
    result = cudf.Series._from_column(as_column(data, dtype=dtype))
    expected = pd.Series(data, dtype=dtype.to_pandas())
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "klass", [cudf.Series, cudf.Index, pd.Series, pd.Index]
)
def test_series_from_named_object_name_priority(klass):
    result = cudf.Series(klass([1], name="a"), name="b")
    assert result.name == "b"


@pytest.mark.parametrize(
    "data",
    [
        {"a": 1, "b": 2, "c": 3},
        cudf.Series([1, 2, 3], index=list("abc")),
        pd.Series([1, 2, 3], index=list("abc")),
    ],
)
def test_series_from_object_with_index_index_arg_reindex(data):
    result = cudf.Series(data, index=list("bca"))
    expected = cudf.Series([2, 3, 1], index=list("bca"))
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        {0: 1, 1: 2, 2: 3},
        cudf.Series([1, 2, 3]),
        cudf.Index([1, 2, 3]),
        pd.Series([1, 2, 3]),
        pd.Index([1, 2, 3]),
        [1, 2, 3],
    ],
)
def test_series_dtype_astypes(data):
    result = cudf.Series(data, dtype="float64")
    expected = cudf.Series([1.0, 2.0, 3.0])
    assert_eq(result, expected)


@pytest.mark.parametrize("pa_type", [pa.string, pa.large_string])
def test_series_from_large_string(pa_type):
    pa_string_array = pa.array(["a", "b", "c"]).cast(pa_type())
    got = cudf.Series(pa_string_array)
    expected = pd.Series(pa_string_array)

    assert_eq(expected, got)


def test_series_init_with_nans():
    with cudf.option_context("mode.pandas_compatible", True):
        gs = cudf.Series([1, 2, 3, np.nan])
    assert gs.dtype == np.dtype("float64")
    ps = pd.Series([1, 2, 3, np.nan])
    assert_eq(ps, gs)


@pytest.mark.parametrize(
    "data",
    [
        [[1, 2, 3], [10, 20]],
        [[1.0, 2.0, 3.0], None, [10.0, 20.0, np.nan]],
        [[5, 6], None, [1]],
        [None, None, None, None, None, [10, 20]],
    ],
)
@pytest.mark.parametrize("klass", [cudf.Series, list, cp.array])
def test_nested_series_from_sequence_data(data, klass):
    actual = cudf.Series(
        [klass(val) if val is not None else val for val in data]
    )
    expected = cudf.Series(data)
    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "data",
    [
        lambda: cp.ones(5, dtype=cp.float16),
        lambda: np.ones(5, dtype="float16"),
        lambda: pd.Series([0.1, 1.2, 3.3], dtype="float16"),
        pytest.param(
            lambda: pa.array(np.ones(5, dtype="float16")),
            marks=pytest.mark.xfail(
                reason="https://issues.apache.org/jira/browse/ARROW-13762"
            ),
        ),
    ],
)
def test_series_raises_float16(data):
    data = data()
    with pytest.raises(TypeError):
        cudf.Series(data)


@pytest.mark.parametrize(
    "data", [[True, False, None, True, False], [None, None], []]
)
@pytest.mark.parametrize("bool_dtype", ["bool", "boolean", pd.BooleanDtype()])
def test_nullable_bool_dtype_series(data, bool_dtype):
    psr = pd.Series(data, dtype=pd.BooleanDtype())
    gsr = cudf.Series(data, dtype=bool_dtype)

    assert_eq(psr, gsr.to_pandas(nullable=True))


@pytest.mark.parametrize("data", [None, 123, 33243243232423, 0])
@pytest.mark.parametrize("klass", [pd.Timestamp, pd.Timedelta])
def test_temporal_scalar_series_init(data, klass):
    scalar = klass(data)
    expected = pd.Series([scalar])
    actual = cudf.Series([scalar])

    assert_eq(expected, actual)

    expected = pd.Series(scalar)
    actual = cudf.Series(scalar)

    assert_eq(expected, actual)


def test_series_from_series_index_no_shallow_copy():
    ser1 = cudf.Series(range(3), index=list("abc"))
    ser2 = cudf.Series(ser1)
    assert ser1.index is ser2.index
