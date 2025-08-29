# Copyright (c) 2025, NVIDIA CORPORATION.

import datetime
import zoneinfo

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


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
    gd_data = cudf.Series.from_pandas(pd_data)

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
    gdf_sr = cudf.Series.from_pandas(pd_sr)

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
    ordered_dtype_gd = cudf.CategoricalDtype.from_pandas(ordered_dtype_pd)
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
    gd_dtype = cudf.CategoricalDtype.from_pandas(pd_dtype)
    gd_to_dtype = cudf.CategoricalDtype.from_pandas(pd_to_dtype)

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
