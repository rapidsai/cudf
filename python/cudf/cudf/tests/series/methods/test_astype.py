# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import datetime
import zoneinfo
from decimal import Decimal

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core.column.decimal import Decimal32Column, Decimal64Column
from cudf.core.column.numerical import NumericalColumn
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


@pytest.mark.parametrize(
    "data",
    [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 3], [4, 5, 6]],
        ],
        [[[1, 2, 3], [4, None, 6]], [], None, [[7, 8], [], None, [9]]],
        [[1, 2, 3], [4, None, 6], [7, 8], [], None, [9]],
        [[1.0, 2.0, 3.0], [4.0, None, 6.0], [7.0, 8.0], [], None, [9.0]],
    ],
)
def test_listcol_as_string(data):
    got = cudf.Series(data).astype("str")
    expect = pd.Series(data).astype("str")
    assert_eq(expect, got)


def test_list_astype():
    s = cudf.Series([[1, 2], [3, 4]])
    s2 = s.list.astype("float64")
    assert s2.dtype == cudf.ListDtype("float64")
    assert_eq(s.list.leaves.astype("float64"), s2.list.leaves)

    s = cudf.Series([[[1, 2], [3]], [[5, 6], None]])
    s2 = s.list.astype("string")
    assert s2.dtype == cudf.ListDtype(cudf.ListDtype("string"))
    assert_eq(s.list.leaves.astype("string"), s2.list.leaves)


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


@pytest.mark.parametrize(
    "alias,expect_dtype",
    [
        ("UInt8", "uint8"),
        ("UInt16", "uint16"),
        ("UInt32", "uint32"),
        ("UInt64", "uint64"),
        ("Int8", "int8"),
        ("Int16", "int16"),
        ("Int32", "int32"),
        ("Int64", "int64"),
        ("boolean", "bool"),
        ("Float32", "float32"),
        ("Float64", "float64"),
    ],
)
def test_astype_with_aliases(alias, expect_dtype):
    pd_data = pd.Series([1, 2, 0])
    gd_data = cudf.Series(pd_data)

    assert_eq(pd_data.astype(expect_dtype), gd_data.astype(alias))


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


def test_typecast_from_datetime(numeric_types_as_str):
    data = pd.date_range(
        "2019-07-16 00:00:00",
        "2019-07-16 00:00:01",
        freq="5555us",
        name="times",
    )
    pd_data = pd.Series(data)
    np_data = np.array(pd_data)
    gdf_data = cudf.Series(pd_data)

    np_casted = np_data.astype(numeric_types_as_str)
    gdf_casted = gdf_data.astype(numeric_types_as_str)

    np.testing.assert_equal(np_casted, gdf_casted.to_numpy())


def test_typecast_from_datetime_to_int64_to_datetime(datetime_types_as_str):
    data = pd.date_range(
        "2019-07-16 00:00:00",
        "2019-07-16 00:00:01",
        freq="5555us",
        name="times",
    )
    pd_data = pd.Series(data)
    np_data = np.array(pd_data)
    gdf_data = cudf.Series(pd_data)

    np_casted = np_data.astype(np.int64).astype(datetime_types_as_str)
    gdf_casted = gdf_data.astype(np.int64).astype(datetime_types_as_str)

    np.testing.assert_equal(np_casted, gdf_casted.to_numpy())


def test_typecast_to_different_datetime_resolutions(datetime_types_as_str):
    data = pd.date_range(
        "2019-07-16 00:00:00",
        "2019-07-16 00:00:01",
        freq="5555us",
        name="times",
    )
    pd_data = pd.Series(data)
    np_data = np.array(pd_data).astype(datetime_types_as_str)
    gdf_series = cudf.Series(pd_data).astype(datetime_types_as_str)
    np.testing.assert_equal(np_data, gdf_series.to_numpy())


@pytest.mark.parametrize(
    "data",
    [
        [
            "2019-07-16 00:00:00.333",
            "2019-07-16 00:00:00.666",
            "2019-07-16 00:00:00.888",
        ],
        [
            "2019-07-16 00:00:00.333333",
            "2019-07-16 00:00:00.666666",
            "2019-07-16 00:00:00.888888",
        ],
        [
            "2019-07-16 00:00:00.333333333",
            "2019-07-16 00:00:00.666666666",
            "2019-07-16 00:00:00.888888888",
        ],
    ],
    ids=["ms_data", "us_data", "ns_data"],
)
def test_string_timstamp_typecast_to_different_datetime_resolutions(
    data, datetime_types_as_str
):
    pd_sr = pd.Series(data)
    gdf_sr = cudf.Series(pd_sr)

    expect = pd_sr.values.astype(datetime_types_as_str)
    got = gdf_sr.astype(datetime_types_as_str).values_host

    np.testing.assert_equal(expect, got)


def test_typecast_to_datetime(numeric_types_as_str, datetime_types_as_str):
    data = np.arange(1, 10)
    np_data = data.astype(numeric_types_as_str)
    gdf_data = cudf.Series(np_data)

    np_casted = np_data.astype(datetime_types_as_str)
    gdf_casted = gdf_data.astype(datetime_types_as_str)

    np.testing.assert_equal(np_casted, gdf_casted.to_numpy())


def test_typecast_to_from_datetime(
    numeric_types_as_str, datetime_types_as_str
):
    data = np.arange(1, 10)
    np_data = data.astype(numeric_types_as_str)
    gdf_data = cudf.Series(np_data)

    np_casted = np_data.astype(datetime_types_as_str).astype(
        numeric_types_as_str
    )
    gdf_casted = gdf_data.astype(datetime_types_as_str).astype(
        numeric_types_as_str
    )

    np.testing.assert_equal(np_casted, gdf_casted.to_numpy())


@pytest.fixture
def datetime_types_as_str2(datetime_types_as_str):
    return datetime_types_as_str


def test_typecast_from_datetime_to_datetime(
    datetime_types_as_str, datetime_types_as_str2
):
    data = np.arange(1, 10)
    np_data = data.astype(datetime_types_as_str)
    ser = cudf.Series(np_data)

    np_casted = np_data.astype(datetime_types_as_str2)
    ser_casted = ser.astype(datetime_types_as_str2)

    np.testing.assert_equal(np_casted, ser_casted.to_numpy())


@pytest.mark.parametrize(
    "data",
    [
        ["2001-01-01", "2002-02-02", "2000-01-05", "NaT"],
        ["2001-01-01", "2002-02-02", "2000-01-05", None],
        [None, None, None, None, None],
    ],
)
def test_str_null_to_datetime(data, datetime_types_as_str):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    assert_eq(
        psr.astype(datetime_types_as_str), gsr.astype(datetime_types_as_str)
    )


def test_str_to_datetime_error():
    psr = pd.Series(["2001-01-01", "2002-02-02", "2000-01-05", "None"])
    gsr = cudf.Series(["2001-01-01", "2002-02-02", "2000-01-05", "None"])

    assert_exceptions_equal(
        lfunc=psr.astype,
        rfunc=gsr.astype,
        lfunc_args_and_kwargs=(["datetime64[s]"],),
        rfunc_args_and_kwargs=(["datetime64[s]"],),
        check_exception_type=False,
    )


@pytest.mark.parametrize("timezone", ["", "Z"])
@pytest.mark.parametrize(
    "data",
    [
        "2002-10-27T04:30",
        "2002-10-27T04:30:00",
        "2002-10-27T04:30:00.000",
        "2002-10-27T04:30:00.000000",
        "2002-10-27T04:30:00.000000000",
    ],
)
def test_datetime_infer_format(data, timezone, datetime_types_as_str):
    ts_data = [data + timezone]
    sr = cudf.Series(ts_data)
    psr = pd.Series(ts_data)
    if not timezone:
        expected = psr.astype(datetime_types_as_str)
        actual = sr.astype(datetime_types_as_str)

        assert_eq(expected, actual)
    else:
        with cudf.option_context("mode.pandas_compatible", True):
            with pytest.raises(NotImplementedError):
                # pandas doesn't allow parsing "Z" to naive type
                sr.astype(datetime_types_as_str)


def test_string_astype(all_supported_types_as_str):
    if all_supported_types_as_str.startswith(
        "int"
    ) or all_supported_types_as_str.startswith("uint"):
        data = ["1", "2", "3", "4", "5"]
    elif all_supported_types_as_str.startswith("float"):
        data = [
            "1.0",
            "2.0",
            "3.0",
            "4.0",
            None,
            "5.0",
            "nan",
            "-INF",
            "NaN",
            "inF",
            "NAn",
        ]
    elif all_supported_types_as_str.startswith("bool"):
        data = ["True", "False", "True", "False", "False"]
    elif all_supported_types_as_str.startswith("datetime64"):
        data = [
            "2019-06-04T00:00:00",
            "2019-06-04T12:12:12",
            "2019-06-03T00:00:00",
            "2019-05-04T00:00:00",
            "2018-06-04T00:00:00",
            "1922-07-21T01:02:03",
        ]
    elif all_supported_types_as_str.startswith("timedelta64"):
        data = [
            "1 days 00:00:00",
            "2 days 00:00:00",
            "3 days 00:00:00",
        ]
    elif all_supported_types_as_str in {"str", "category"}:
        data = ["ab", "cd", "ef", "gh", "ij"]
    ps = pd.Series(data)
    gs = cudf.Series(data)

    expect = ps.astype(all_supported_types_as_str)
    got = gs.astype(all_supported_types_as_str)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data, scale, precision",
    [
        (["1.11", "2.22", "3.33"], 2, 3),
        (["111", "222", "33"], 0, 3),
        (["111000", "22000", "3000"], -3, 3),
        ([None, None, None], 0, 5),
        ([None, "-2345", None], 0, 5),
        ([], 0, 5),
    ],
)
@pytest.mark.parametrize(
    "decimal_dtype",
    [cudf.Decimal128Dtype, cudf.Decimal64Dtype, cudf.Decimal32Dtype],
)
def test_string_to_decimal(data, scale, precision, decimal_dtype):
    gs = cudf.Series(data, dtype="str")
    fp = gs.astype(decimal_dtype(scale=scale, precision=precision))
    got = fp.astype("str")
    assert_eq(gs, got)


def test_string_empty_to_decimal():
    gs = cudf.Series(["", "-85", ""], dtype="str")
    got = gs.astype(cudf.Decimal64Dtype(scale=0, precision=5))
    expected = cudf.Series(
        [0, -85, 0],
        dtype=cudf.Decimal64Dtype(scale=0, precision=5),
    )
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data, scale, precision",
    [
        (["1.23", "-2.34", "3.45"], 2, 3),
        (["123", "-234", "345"], 0, 3),
        (["12300", "-400", "5000.0"], -2, 5),
        ([None, None, None], 0, 5),
        ([None, "-100", None], 0, 5),
        ([], 0, 5),
    ],
)
@pytest.mark.parametrize(
    "decimal_dtype",
    [cudf.Decimal128Dtype, cudf.Decimal32Dtype, cudf.Decimal64Dtype],
)
def test_string_from_decimal(data, scale, precision, decimal_dtype):
    decimal_data = []
    for d in data:
        if d is None:
            decimal_data.append(None)
        else:
            decimal_data.append(Decimal(d))
    fp = cudf.Series(
        decimal_data,
        dtype=decimal_dtype(scale=scale, precision=precision),
    )
    gs = fp.astype("str")
    got = gs.astype(decimal_dtype(scale=scale, precision=precision))
    assert_eq(fp, got)


def test_string_empty_astype(all_supported_types_as_str):
    data = []
    ps = pd.Series(data, dtype="str")
    gs = cudf.Series(data, dtype="str")

    expect = ps.astype(all_supported_types_as_str)
    got = gs.astype(all_supported_types_as_str)

    assert_eq(expect, got)


def test_string_numeric_astype(numeric_and_temporal_types_as_str):
    if numeric_and_temporal_types_as_str.startswith("timedelta64"):
        pytest.skip(
            f"Test not applicable for {numeric_and_temporal_types_as_str}"
        )
    if numeric_and_temporal_types_as_str.startswith("bool"):
        data = [1, 0, 1, 0, 1]
    elif numeric_and_temporal_types_as_str.startswith(
        "int"
    ) or numeric_and_temporal_types_as_str.startswith("uint"):
        data = [1, 2, 3, 4, 5]
    elif numeric_and_temporal_types_as_str.startswith("float"):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
    elif numeric_and_temporal_types_as_str.startswith("datetime64"):
        # pandas rounds the output format based on the data
        # Use numpy instead
        # but fix '2011-01-01T00:00:00' -> '2011-01-01 00:00:00'
        data = [1000000001, 2000000001, 3000000001, 4000000001, 5000000001]
        ps = np.asarray(data, dtype=numeric_and_temporal_types_as_str).astype(
            str
        )
        ps = np.array([i.replace("T", " ") for i in ps])

    if not numeric_and_temporal_types_as_str.startswith("datetime64"):
        ps = pd.Series(data, dtype=numeric_and_temporal_types_as_str)

    gs = cudf.Series(data, dtype=numeric_and_temporal_types_as_str)

    expect = pd.Series(ps.astype("str"))
    got = gs.astype("str")

    assert_eq(expect, got)


def test_string_empty_numeric_astype(numeric_and_temporal_types_as_str):
    data = []

    ps = pd.Series(data, dtype=numeric_and_temporal_types_as_str)
    gs = cudf.Series(data, dtype=numeric_and_temporal_types_as_str)

    expect = ps.astype("str")
    got = gs.astype("str")

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data,dtype",
    [
        (["0.1", "10.2", "10.876"], "float"),
        (["-0.1", "10.2", "+10.876"], "float"),
        (["1", "10.2", "10.876"], "float32"),
        (["+123", "6344556789", "0"], "int"),
        (["+123", "6344556789", "0"], "uint64"),
        (["+123", "6344556789", "0"], "float"),
        (["0.1", "-10.2", "10.876", None], "float"),
    ],
)
@pytest.mark.parametrize("obj_type", [None, "str", "category"])
def test_string_typecast(data, obj_type, dtype):
    psr = pd.Series(data, dtype=obj_type)
    gsr = cudf.Series(data, dtype=obj_type)

    expect = psr.astype(dtype=dtype)
    actual = gsr.astype(dtype=dtype)
    assert_eq(expect, actual)


@pytest.mark.parametrize(
    "data,dtype",
    [
        (["0.1", "10.2", "10.876"], "int"),
        (["1", "10.2", "+10.876"], "int"),
        (["abc", "1", "2", " "], "int"),
        (["0.1", "10.2", "10.876"], "uint64"),
        (["1", "10.2", "+10.876"], "uint64"),
        (["abc", "1", "2", " "], "uint64"),
        ([" ", "0.1", "2"], "float"),
        ([""], "int"),
        ([""], "uint64"),
        ([" "], "float"),
        (["\n"], "int"),
        (["\n"], "uint64"),
        (["0.1", "-10.2", "10.876", None], "int"),
        (["0.1", "-10.2", "10.876", None], "uint64"),
        (["0.1", "-10.2", "10.876", None, "ab"], "float"),
        (["+", "-"], "float"),
        (["+", "-"], "int"),
        (["+", "-"], "uint64"),
        (["1++++", "--2"], "float"),
        (["1++++", "--2"], "int"),
        (["1++++", "--2"], "uint64"),
        (["++++1", "--2"], "float"),
        (["++++1", "--2"], "int"),
        (["++++1", "--2"], "uint64"),
    ],
)
@pytest.mark.parametrize("obj_type", [None, "str", "category"])
def test_string_typecast_error(data, obj_type, dtype):
    psr = pd.Series(data, dtype=obj_type)
    gsr = cudf.Series(data, dtype=obj_type)

    assert_exceptions_equal(
        lfunc=psr.astype,
        rfunc=gsr.astype,
        lfunc_args_and_kwargs=([dtype],),
        rfunc_args_and_kwargs=([dtype],),
    )


@pytest.mark.parametrize("unit", ["ns", "us"])
def test_astype_aware_to_aware(unit):
    ser = cudf.Series(
        [datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)]
    )
    result = ser.astype(f"datetime64[{unit}, US/Pacific]")
    expected = ser.to_pandas().astype(f"datetime64[{unit}, US/Pacific]")
    zoneinfo_type = pd.DatetimeTZDtype(
        expected.dtype.unit, zoneinfo.ZoneInfo(str(expected.dtype.tz))
    )
    expected = ser.astype(zoneinfo_type)
    assert_eq(result, expected)


def test_astype_naive_to_aware_raises():
    ser = cudf.Series([datetime.datetime(2020, 1, 1)])
    with pytest.raises(TypeError):
        ser.astype("datetime64[ns, UTC]")
    with pytest.raises(TypeError):
        ser.to_pandas().astype("datetime64[ns, UTC]")


@pytest.mark.parametrize(
    "np_dtype,pd_dtype",
    [
        tuple(item)
        for item in cudf.utils.dtypes.np_dtypes_to_pandas_dtypes.items()
    ],
)
def test_series_astype_pandas_nullable(
    all_supported_types_as_str, np_dtype, pd_dtype
):
    source = cudf.Series([0, 1, None], dtype=all_supported_types_as_str)

    expect = source.astype(np_dtype)
    got = source.astype(pd_dtype)

    assert_eq(expect, got)


def test_series_astype_numeric_to_numeric(
    numeric_types_as_str, numeric_types_as_str2
):
    psr = pd.Series([1, 2, 4, 3], dtype=numeric_types_as_str)
    gsr = cudf.from_pandas(psr)
    assert_eq(
        psr.astype(numeric_types_as_str2), gsr.astype(numeric_types_as_str2)
    )


def test_series_astype_numeric_to_numeric_nulls(
    numeric_types_as_str, numeric_types_as_str2
):
    data = [1, 2, None, 3]
    sr = cudf.Series(data, dtype=numeric_types_as_str)
    got = sr.astype(numeric_types_as_str2)
    expect = cudf.Series([1, 2, None, 3], dtype=numeric_types_as_str2)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "as_dtype",
    [
        "str",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_series_astype_numeric_to_other(
    request, all_supported_types_as_str, as_dtype
):
    if all_supported_types_as_str.startswith(
        "timedelta64"
    ) and as_dtype.startswith("datetime64"):
        pytest.skip(
            f"Casting {all_supported_types_as_str} to {as_dtype} is invalid"
        )
    if all_supported_types_as_str == "str" and as_dtype.startswith(
        "datetime64"
    ):
        pytest.skip(
            f"Casting {all_supported_types_as_str} to {as_dtype} for test data is invalid."
        )
    request.applymarker(
        pytest.mark.xfail(
            all_supported_types_as_str
            in {"timedelta64[us]", "timedelta64[ms]", "timedelta64[s]"}
            and as_dtype == "str",
            reason=f"Casting {all_supported_types_as_str} to {as_dtype} is incorrect.",
        )
    )
    psr = pd.Series([1, 2, 3], dtype=all_supported_types_as_str)
    gsr = cudf.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype))


@pytest.mark.parametrize(
    "as_dtype",
    [
        "str",
        "int32",
        "uint32",
        "float32",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_series_astype_string_to_other(as_dtype):
    if "datetime64" in as_dtype:
        data = ["2001-01-01", "2002-02-02", "2000-01-05"]
    else:
        data = ["1", "2", "3"]
    psr = pd.Series(data)
    gsr = cudf.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype))


@pytest.mark.parametrize(
    "as_dtype",
    [
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_series_astype_datetime_to_other(as_dtype):
    data = ["2001-01-01", "2002-02-02", "2001-01-05"]
    psr = pd.Series(data)
    gsr = cudf.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype))


@pytest.mark.parametrize(
    "inp",
    [
        ("datetime64[ns]", "2011-01-01 00:00:00.000000000"),
        ("datetime64[us]", "2011-01-01 00:00:00.000000"),
        ("datetime64[ms]", "2011-01-01 00:00:00.000"),
        ("datetime64[s]", "2011-01-01 00:00:00"),
    ],
)
def test_series_astype_datetime_to_string(inp):
    dtype, expect = inp
    base_date = "2011-01-01"
    sr = cudf.Series([base_date], dtype=dtype)
    got = sr.astype(str)[0]
    assert expect == got


@pytest.mark.parametrize(
    "as_dtype",
    [
        "int32",
        "uint32",
        "float32",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
        "str",
    ],
)
def test_series_astype_categorical_to_other(as_dtype):
    if "datetime64" in as_dtype:
        data = ["2001-01-01", "2002-02-02", "2000-01-05", "2001-01-01"]
    else:
        data = [1, 2, 3, 1]
    psr = pd.Series(data, dtype="category")
    gsr = cudf.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype))


def test_series_astype_to_categorical_ordered(categorical_ordered):
    psr = pd.Series([1, 2, 3, 1], dtype="category")
    gsr = cudf.from_pandas(psr)

    ordered_dtype_pd = pd.CategoricalDtype(
        categories=[1, 2, 3], ordered=categorical_ordered
    )
    ordered_dtype_gd = cudf.CategoricalDtype(
        categories=ordered_dtype_pd.categories,
        ordered=ordered_dtype_pd.ordered,
    )
    assert_eq(
        psr.astype("int32").astype(ordered_dtype_pd).astype("int32"),
        gsr.astype("int32").astype(ordered_dtype_gd).astype("int32"),
    )


def test_series_astype_cat_ordered_to_unordered(categorical_ordered):
    pd_dtype = pd.CategoricalDtype(
        categories=[1, 2, 3], ordered=categorical_ordered
    )
    pd_to_dtype = pd.CategoricalDtype(
        categories=[1, 2, 3], ordered=not categorical_ordered
    )
    gd_dtype = cudf.CategoricalDtype(
        categories=pd_dtype.categories,
        ordered=pd_dtype.ordered,
    )
    gd_to_dtype = cudf.CategoricalDtype(
        categories=pd_to_dtype.categories,
        ordered=pd_to_dtype.ordered,
    )

    psr = pd.Series([1, 2, 3], dtype=pd_dtype)
    gsr = cudf.Series([1, 2, 3], dtype=gd_dtype)

    expect = psr.astype(pd_to_dtype)
    got = gsr.astype(gd_to_dtype)

    assert_eq(expect, got)


def test_series_astype_null_cases():
    data = [1, 2, None, 3]

    # numerical to other
    assert_eq(cudf.Series(data, dtype="str"), cudf.Series(data).astype("str"))

    assert_eq(
        cudf.Series(data, dtype="category"),
        cudf.Series(data).astype("category"),
    )

    assert_eq(
        cudf.Series(data, dtype="float32"),
        cudf.Series(data, dtype="int32").astype("float32"),
    )

    assert_eq(
        cudf.Series(data, dtype="float32"),
        cudf.Series(data, dtype="uint32").astype("float32"),
    )

    assert_eq(
        cudf.Series(data, dtype="datetime64[ms]"),
        cudf.Series(data).astype("datetime64[ms]"),
    )

    # categorical to other
    assert_eq(
        cudf.Series(data, dtype="str"),
        cudf.Series(data, dtype="category").astype("str"),
    )

    assert_eq(
        cudf.Series(data, dtype="float32"),
        cudf.Series(data, dtype="category").astype("float32"),
    )

    assert_eq(
        cudf.Series(data, dtype="datetime64[ms]"),
        cudf.Series(data, dtype="category").astype("datetime64[ms]"),
    )

    # string to other
    assert_eq(
        cudf.Series([1, 2, None, 3], dtype="int32"),
        cudf.Series(["1", "2", None, "3"]).astype("int32"),
    )

    assert_eq(
        cudf.Series(
            ["2001-01-01", "2001-02-01", None, "2001-03-01"],
            dtype="datetime64[ms]",
        ),
        cudf.Series(["2001-01-01", "2001-02-01", None, "2001-03-01"]).astype(
            "datetime64[ms]"
        ),
    )

    assert_eq(
        cudf.Series(["a", "b", "c", None], dtype="category").to_pandas(),
        cudf.Series(["a", "b", "c", None]).astype("category").to_pandas(),
    )

    # datetime to other
    data = [
        "2001-01-01 00:00:00.000000",
        "2001-02-01 00:00:00.000000",
        None,
        "2001-03-01 00:00:00.000000",
    ]
    assert_eq(
        cudf.Series(data),
        cudf.Series(data, dtype="datetime64[us]").astype("str"),
    )

    assert_eq(
        pd.Series(data, dtype="datetime64[ns]").astype("category"),
        cudf.from_pandas(pd.Series(data, dtype="datetime64[ns]")).astype(
            "category"
        ),
    )


def test_series_astype_null_categorical():
    sr = cudf.Series([None, None, None], dtype="category")
    expect = cudf.Series([None, None, None], dtype="int32")
    got = sr.astype("int32")
    assert_eq(expect, got)


@pytest.mark.parametrize("precision, scale", [(7, 2), (11, 4), (18, 9)])
def test_typecast_from_float_to_decimal(
    request, float_types_as_str, precision, scale
):
    to_dtype = cudf.Decimal64Dtype(precision, scale)
    data = cudf.Series(
        [
            14.12302,
            97938.2,
            np.nan,
            0.0,
            -8.302014,
            np.nan,
            94.31304,
            -112.2314,
            0.3333333,
            np.nan,
        ]
    )
    request.applymarker(
        pytest.mark.xfail(
            float_types_as_str == "float32" and to_dtype.precision > 12,
            reason="https://github.com/rapidsai/cudf/issues/14169",
        )
    )
    got = data.astype(float_types_as_str)

    pa_arr = got.to_arrow().cast(
        pa.decimal128(to_dtype.precision, to_dtype.scale)
    )
    expected = cudf.Series._from_column(Decimal64Column.from_arrow(pa_arr))

    got = got.astype(to_dtype)

    assert_eq(got, expected)


@pytest.mark.parametrize("precision, scale", [(9, 3), (11, 4), (18, 9)])
def test_typecast_from_int_to_decimal(integer_types_as_str, precision, scale):
    to_dtype = cudf.Decimal64Dtype(precision, scale)
    data = cudf.Series(
        [
            14.12302,
            38.2,
            np.nan,
            0.0,
            -8.302014,
            np.nan,
            94.31304,
            np.nan,
            -112.2314,
            0.3333333,
            np.nan,
        ]
    )
    got = data.astype(integer_types_as_str)

    pa_arr = (
        got.to_arrow()
        .cast("float64")
        .cast(pa.decimal128(to_dtype.precision, to_dtype.scale))
    )
    expected = cudf.Series._from_column(Decimal64Column.from_arrow(pa_arr))

    got = got.astype(to_dtype)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "from_dtype",
    [
        cudf.Decimal64Dtype(7, 2),
        cudf.Decimal64Dtype(11, 4),
        cudf.Decimal64Dtype(18, 10),
        cudf.Decimal32Dtype(7, 2),
        cudf.Decimal32Dtype(5, 3),
        cudf.Decimal32Dtype(9, 5),
    ],
)
@pytest.mark.parametrize(
    "to_dtype",
    [
        cudf.Decimal64Dtype(7, 2),
        cudf.Decimal64Dtype(18, 10),
        cudf.Decimal64Dtype(11, 4),
        cudf.Decimal32Dtype(7, 2),
        cudf.Decimal32Dtype(9, 5),
        cudf.Decimal32Dtype(5, 3),
    ],
)
def test_typecast_to_from_decimal(from_dtype, to_dtype):
    data = cudf.Series(
        [
            14.12309,
            2.343942,
            np.nan,
            0.0,
            -8.302082,
            np.nan,
            94.31308,
            -112.2364,
            -8.029972,
            np.nan,
        ]
    )
    if from_dtype.scale > to_dtype.MAX_PRECISION:
        pytest.skip(
            "This is supposed to overflow because the representation value in "
            "the source exceeds the max representable in destination dtype."
        )
    s = data.astype(from_dtype)

    pa_arr = s.to_arrow().cast(
        pa.decimal128(to_dtype.precision, to_dtype.scale), safe=False
    )
    if isinstance(to_dtype, cudf.Decimal32Dtype):
        expected = cudf.Series._from_column(Decimal32Column.from_arrow(pa_arr))
    elif isinstance(to_dtype, cudf.Decimal64Dtype):
        expected = cudf.Series._from_column(Decimal64Column.from_arrow(pa_arr))

    with expect_warning_if(to_dtype.scale < s.dtype.scale, UserWarning):
        got = s.astype(to_dtype)

    assert_eq(got, expected)


@pytest.mark.parametrize("precision, scale", [(7, 2), (11, 4), (17, 10)])
def test_typecast_from_decimal(precision, scale, signed_integer_types_as_str):
    from_dtype = cudf.Decimal64Dtype(precision, scale)
    data = cudf.Series(
        [
            14.12309,
            2.343942,
            np.nan,
            0.0,
            -8.302082,
            np.nan,
            94.31308,
            -112.2364,
            -8.029972,
            np.nan,
        ]
    )
    got = data.astype(from_dtype)
    pa_arr = got.to_arrow().cast(signed_integer_types_as_str, safe=False)

    got = got.astype(signed_integer_types_as_str)
    expected = cudf.Series._from_column(NumericalColumn.from_arrow(pa_arr))

    assert_eq(got, expected)
    assert_eq(got.dtype, expected.dtype)


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([1, 2, 3, 89]),
        pd.Series([1, 2, 3, 89, 3, 1, 89], dtype="category"),
        pd.Series(["1", "2", "3", "4", "5"], dtype="category"),
        pd.Series(["1.0", "2.5", "3.001", "9"], dtype="category"),
        pd.Series(["1", "2", "3", None, "4", "5"], dtype="category"),
        pd.Series(["1.0", "2.5", "3.001", None, "9"], dtype="category"),
        pd.Series(["a", "b", "c", "c", "b", "a", "b", "b"]),
        pd.Series(["aa", "b", "c", "c", "bb", "bb", "a", "b", "b"]),
        pd.Series([1, 2, 3, 89, None, np.nan, np.nan], dtype="float64"),
        pd.Series([1, 2, 3, 89], dtype="float64"),
        pd.Series([1, 2.5, 3.001, 89], dtype="float64"),
        pd.Series([None, None, None]),
        pd.Series([], dtype="float64"),
    ],
)
@pytest.mark.parametrize(
    "categories",
    [
        ["aa", "bb", "cc"],
        [2, 4, 10, 100],
        ["aa", "bb", "c"],
        ["a", "bb", "c"],
        ["a", "b", "c"],
        ["1", "2", "3", "4"],
        ["1.0", "2.5", "3.001", "9"],
        [],
    ],
)
def test_categorical_typecast(data, categories):
    pd_data = data
    gd_data = cudf.from_pandas(data)
    cat_type = pd.CategoricalDtype(categories)

    assert_eq(pd_data.astype(cat_type), gd_data.astype(cat_type))


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1], np.uint8),
        ([1, None], np.uint8),
        (np.arange(np.iinfo(np.int8).max), np.uint8),
        (np.append(np.arange(np.iinfo(np.int8).max), [None]), np.uint8),
        (np.arange(np.iinfo(np.int16).max), np.uint16),
        (np.append(np.arange(np.iinfo(np.int16).max), [None]), np.uint16),
        (np.arange(np.iinfo(np.uint8).max), np.uint8),
        (np.append(np.arange(np.iinfo(np.uint8).max), [None]), np.uint8),
        (np.arange(np.iinfo(np.uint16).max), np.uint16),
        (np.append(np.arange(np.iinfo(np.uint16).max), [None]), np.uint16),
    ],
)
def test_astype_dtype(values, expected):
    data = cudf.Series(values)
    got = data.astype("category").cat.codes.dtype
    np.testing.assert_equal(got, expected)


@pytest.mark.parametrize("ordered", [True, False])
def test_empty_series_category_cast(ordered):
    dtype = cudf.CategoricalDtype(ordered=ordered)
    ps = pd.Series([], dtype="str")
    gs = cudf.from_pandas(ps)

    expected = ps.astype(dtype.to_pandas())
    actual = gs.astype(dtype)

    assert_eq(expected, actual)
    assert_eq(expected.dtype.ordered, actual.dtype.ordered)
