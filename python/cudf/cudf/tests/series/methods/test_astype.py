# Copyright (c) 2025, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


def test_series_typecast_to_object_error():
    actual = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(TypeError):
            actual.astype(object)
        with pytest.raises(TypeError):
            actual.astype(np.dtype("object"))
        new_series = actual.astype("str")
        assert new_series[0] == "1970-01-01 00:00:00.000000001"


def test_series_typecast_to_object():
    actual = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    with cudf.option_context("mode.pandas_compatible", False):
        new_series = actual.astype(object)
        assert new_series[0] == "1970-01-01 00:00:00.000000001"
        new_series = actual.astype(np.dtype("object"))
        assert new_series[0] == "1970-01-01 00:00:00.000000001"


@pytest.mark.parametrize(
    "dtype",
    [
        pd.ArrowDtype(pa.int8()),
        pd.ArrowDtype(pa.int16()),
        pd.ArrowDtype(pa.int32()),
        pd.ArrowDtype(pa.int64()),
        pd.ArrowDtype(pa.uint8()),
        pd.ArrowDtype(pa.uint16()),
        pd.ArrowDtype(pa.uint32()),
        pd.ArrowDtype(pa.uint64()),
        pd.ArrowDtype(pa.float32()),
        pd.ArrowDtype(pa.float64()),
        pd.Int8Dtype(),
        pd.Int16Dtype(),
        pd.Int32Dtype(),
        pd.Int64Dtype(),
        pd.UInt8Dtype(),
        pd.UInt16Dtype(),
        pd.UInt32Dtype(),
        pd.UInt64Dtype(),
        pd.Float32Dtype(),
        pd.Float64Dtype(),
    ],
)
@pytest.mark.parametrize("klass", [cudf.Series, cudf.DataFrame, cudf.Index])
@pytest.mark.parametrize("kind", [lambda x: x, str], ids=["obj", "string"])
def test_astype_pandas_nullable_pandas_compat(dtype, klass, kind):
    ser = klass([1, 2, 3])
    with cudf.option_context("mode.pandas_compatible", True):
        actual = ser.astype(kind(dtype))
        expected = klass([1, 2, 3], dtype=kind(dtype))
        assert_eq(actual, expected)


@pytest.mark.parametrize(
    "type1",
    [
        "category",
        "interval[int64, right]",
        "int64",
        "float64",
        "str",
        "datetime64[ns]",
        "timedelta64[ns]",
    ],
)
@pytest.mark.parametrize(
    "type2",
    [
        "category",
        "interval[int64, right]",
        "int64",
        "float64",
        "str",
        "datetime64[ns]",
        "timedelta64[ns]",
    ],
)
@pytest.mark.parametrize(
    "as_dtype", [lambda x: x, cudf.dtype], ids=["string", "object"]
)
@pytest.mark.parametrize("copy", [True, False])
def test_empty_astype_always_castable(type1, type2, as_dtype, copy):
    ser = cudf.Series([], dtype=as_dtype(type1))
    result = ser.astype(as_dtype(type2), copy=copy)
    expected = cudf.Series([], dtype=as_dtype(type2))
    assert_eq(result, expected)
    if not copy and cudf.dtype(type1) == cudf.dtype(type2):
        assert ser._column is result._column
    else:
        assert ser._column is not result._column


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [12, 12, 22, 343, 4353534, 435342],
        [0.3534, 12, 22, 343, 43.53534, 4353.42],
        cp.asarray([10, 20, 30, 100]),
    ],
)
@pytest.mark.parametrize("cast_dtype", ["int64", "category"])
def test_timedelta_from_typecast(data, timedelta_types_as_str, cast_dtype):
    if timedelta_types_as_str != "timedelta64[ns]":
        pytest.skip(
            "Bug in pandas : https://github.com/pandas-dev/pandas/issues/35465"
        )
    psr = pd.Series(
        cp.asnumpy(data) if isinstance(data, cp.ndarray) else data,
        dtype=timedelta_types_as_str,
    )
    gsr = cudf.Series(data, dtype=timedelta_types_as_str)

    if cast_dtype == "int64":
        assert_eq(psr.values.view(cast_dtype), gsr.astype(cast_dtype).values)
    else:
        assert_eq(psr.astype(cast_dtype), gsr.astype(cast_dtype))


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [12, 12, 22, 343, 4353534, 435342],
        [0.3534, 12, 22, 343, 43.53534, 4353.42],
        cp.asarray([10, 20, 30, 100]),
    ],
)
def test_timedelta_to_typecast(data, timedelta_types_as_str):
    psr = pd.Series(cp.asnumpy(data) if isinstance(data, cp.ndarray) else data)
    gsr = cudf.Series(data)

    assert_eq(
        psr.astype(timedelta_types_as_str), gsr.astype(timedelta_types_as_str)
    )


@pytest.mark.parametrize("data", [[], [1, 2, 3, 4, 5]])
def test_numeric_to_timedelta(
    data, numeric_types_as_str, timedelta_types_as_str
):
    sr = cudf.Series(data, dtype=numeric_types_as_str)
    psr = sr.to_pandas()

    actual = sr.astype(timedelta_types_as_str)
    expected = psr.astype(timedelta_types_as_str)

    assert_eq(expected, actual)


def test_timedelta_datetime_cast_invalid():
    sr = cudf.Series([1, 2, 3], dtype="timedelta64[ns]")
    psr = sr.to_pandas()

    assert_exceptions_equal(
        psr.astype,
        sr.astype,
        (["datetime64[ns]"],),
        (["datetime64[ns]"],),
    )

    sr = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    psr = sr.to_pandas()

    assert_exceptions_equal(
        psr.astype,
        sr.astype,
        (["timedelta64[ns]"],),
        (["timedelta64[ns]"],),
    )


@pytest.mark.parametrize(
    "sr_data, sr_dtype, exp_data, exp_dtype",
    [
        [
            [1, 2, 3],
            "timedelta64[ns]",
            [
                "0 days 00:00:00.000000001",
                "0 days 00:00:00.000000002",
                "0 days 00:00:00.000000003",
            ],
            None,
        ],
        [
            [1000000, 200000, 3000000],
            "timedelta64[ms]",
            ["0 days 00:16:40", "0 days 00:03:20", "0 days 00:50:00"],
            None,
        ],
        [
            [1000000, 200000, 3000000],
            "timedelta64[s]",
            ["11 days 13:46:40", "2 days 07:33:20", "34 days 17:20:00"],
            None,
        ],
        [
            [None, None, None, None, None],
            "timedelta64[us]",
            [None, None, None, None, None],
            "str",
        ],
        [
            [
                136457654,
                None,
                245345345,
                223432411,
                None,
                3634548734,
                23234,
            ],
            "timedelta64[us]",
            [
                "0 days 00:02:16.457654",
                None,
                "0 days 00:04:05.345345",
                "0 days 00:03:43.432411",
                None,
                "0 days 01:00:34.548734",
                "0 days 00:00:00.023234",
            ],
            None,
        ],
        [
            [
                136457654,
                None,
                245345345,
                223432411,
                None,
                3634548734,
                23234,
            ],
            "timedelta64[ms]",
            [
                "1 days 13:54:17.654",
                None,
                "2 days 20:09:05.345",
                "2 days 14:03:52.411",
                None,
                "42 days 01:35:48.734",
                "0 days 00:00:23.234",
            ],
            None,
        ],
        [
            [
                136457654,
                None,
                245345345,
                223432411,
                None,
                3634548734,
                23234,
            ],
            "timedelta64[s]",
            [
                "1579 days 08:54:14",
                None,
                "2839 days 15:29:05",
                "2586 days 00:33:31",
                None,
                "42066 days 12:52:14",
                "0 days 06:27:14",
            ],
            None,
        ],
        [
            [
                136457654,
                None,
                245345345,
                223432411,
                None,
                3634548734,
                23234,
            ],
            "timedelta64[ns]",
            [
                "0 days 00:00:00.136457654",
                None,
                "0 days 00:00:00.245345345",
                "0 days 00:00:00.223432411",
                None,
                "0 days 00:00:03.634548734",
                "0 days 00:00:00.000023234",
            ],
            None,
        ],
    ],
)
def test_timedelta_str_roundtrip(sr_data, sr_dtype, exp_data, exp_dtype):
    gsr = cudf.Series(sr_data, dtype=sr_dtype)
    actual_series = gsr.astype("str")

    expected_series = cudf.Series(exp_data, dtype=exp_dtype)
    assert_eq(expected_series, actual_series)

    assert_eq(gsr, actual_series.astype(gsr.dtype))
