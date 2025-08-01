# Copyright (c) 2019-2025, NVIDIA CORPORATION.

import datetime
import operator

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf import DataFrame, Series
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_GE_230,
    PANDAS_VERSION,
)
from cudf.testing import assert_eq
from cudf.testing._utils import (
    DATETIME_TYPES,
    NUMERIC_TYPES,
    assert_exceptions_equal,
    expect_warning_if,
)


@pytest.fixture(
    params=[
        operator.lt,
        operator.gt,
        operator.le,
        operator.ge,
        operator.eq,
        operator.ne,
    ]
)
def op(request):
    return request.param


@pytest.fixture(
    params=[
        pd.date_range("20010101", "20020215", freq="400h", name="times"),
        pd.date_range(
            "20010101", freq="243434324423423234ns", name="times", periods=10
        ),
    ]
)
def data(request):
    return request.param


@pytest.mark.parametrize(
    "lhs_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
@pytest.mark.parametrize(
    "rhs_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_datetime_series_binops_pandas(lhs_dtype, rhs_dtype):
    pd_data_1 = pd.Series(
        pd.date_range("20010101", "20020215", freq="400h", name="times")
    )
    pd_data_2 = pd.Series(
        pd.date_range("20010101", "20020215", freq="401h", name="times")
    )
    gdf_data_1 = Series(pd_data_1).astype(lhs_dtype)
    gdf_data_2 = Series(pd_data_2).astype(rhs_dtype)
    assert_eq(pd_data_1, gdf_data_1.astype("datetime64[ns]"))
    assert_eq(pd_data_2, gdf_data_2.astype("datetime64[ns]"))
    assert_eq(pd_data_1 < pd_data_2, gdf_data_1 < gdf_data_2)
    assert_eq(pd_data_1 > pd_data_2, gdf_data_1 > gdf_data_2)
    assert_eq(pd_data_1 == pd_data_2, gdf_data_1 == gdf_data_2)
    assert_eq(pd_data_1 <= pd_data_2, gdf_data_1 <= gdf_data_2)
    assert_eq(pd_data_1 >= pd_data_2, gdf_data_1 >= gdf_data_2)


@pytest.mark.parametrize(
    "lhs_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
@pytest.mark.parametrize(
    "rhs_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_datetime_series_binops_numpy(lhs_dtype, rhs_dtype):
    pd_data_1 = pd.Series(
        pd.date_range("20010101", "20020215", freq="400h", name="times")
    )
    pd_data_2 = pd.Series(
        pd.date_range("20010101", "20020215", freq="401h", name="times")
    )
    gdf_data_1 = Series(pd_data_1).astype(lhs_dtype)
    gdf_data_2 = Series(pd_data_2).astype(rhs_dtype)
    np_data_1 = np.array(pd_data_1).astype(lhs_dtype)
    np_data_2 = np.array(pd_data_2).astype(rhs_dtype)
    np.testing.assert_equal(np_data_1, gdf_data_1.to_numpy())
    np.testing.assert_equal(np_data_2, gdf_data_2.to_numpy())
    np.testing.assert_equal(
        np.less(np_data_1, np_data_2), (gdf_data_1 < gdf_data_2).to_numpy()
    )
    np.testing.assert_equal(
        np.greater(np_data_1, np_data_2), (gdf_data_1 > gdf_data_2).to_numpy()
    )
    np.testing.assert_equal(
        np.equal(np_data_1, np_data_2), (gdf_data_1 == gdf_data_2).to_numpy()
    )
    np.testing.assert_equal(
        np.less_equal(np_data_1, np_data_2),
        (gdf_data_1 <= gdf_data_2).to_numpy(),
    )
    np.testing.assert_equal(
        np.greater_equal(np_data_1, np_data_2),
        (gdf_data_1 >= gdf_data_2).to_numpy(),
    )


def test_dt_ops(data):
    pd_data = pd.Series(data)
    gdf_data = Series(data)

    assert_eq(pd_data == pd_data, gdf_data == gdf_data)
    assert_eq(pd_data < pd_data, gdf_data < gdf_data)
    assert_eq(pd_data > pd_data, gdf_data > gdf_data)


@pytest.mark.parametrize("from_dtype", NUMERIC_TYPES)
@pytest.mark.parametrize(
    "to_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_typecast_to_datetime(from_dtype, to_dtype):
    data = np.arange(1, 10)
    np_data = data.astype(from_dtype)
    gdf_data = Series(np_data)

    np_casted = np_data.astype(to_dtype)
    gdf_casted = gdf_data.astype(to_dtype)

    np.testing.assert_equal(np_casted, gdf_casted.to_numpy())


@pytest.mark.parametrize("from_dtype", NUMERIC_TYPES)
@pytest.mark.parametrize(
    "to_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_typecast_to_from_datetime(from_dtype, to_dtype):
    data = np.arange(1, 10)
    np_data = data.astype(from_dtype)
    gdf_data = Series(np_data)

    np_casted = np_data.astype(to_dtype).astype(from_dtype)
    gdf_casted = gdf_data.astype(to_dtype).astype(from_dtype)

    np.testing.assert_equal(np_casted, gdf_casted.to_numpy())


@pytest.mark.parametrize(
    "from_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
@pytest.mark.parametrize(
    "to_dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_typecast_from_datetime_to_datetime(from_dtype, to_dtype):
    data = np.arange(1, 10)
    np_data = data.astype(from_dtype)
    ser = Series(np_data)

    np_casted = np_data.astype(to_dtype)
    ser_casted = ser.astype(to_dtype)

    np.testing.assert_equal(np_casted, ser_casted.to_numpy())


@pytest.mark.parametrize("nulls", ["some", "all"])
def test_to_from_pandas_nulls(data, nulls):
    data = np.arange(1, 10)
    pd_data = pd.Series(data.astype("datetime64[ns]"))
    if nulls == "some":
        # Fill half the values with NaT
        pd_data[list(range(0, len(pd_data), 2))] = np.datetime64("nat", "ns")
    elif nulls == "all":
        # Fill all the values with NaT
        pd_data[:] = np.datetime64("nat", "ns")
    gdf_data = Series.from_pandas(pd_data)

    expect = pd_data
    got = gdf_data.to_pandas()

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "dtype",
    ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
)
def test_datetime_to_arrow(dtype):
    timestamp = (
        cudf.datasets.timeseries(
            start="2000-01-01", end="2000-01-02", freq="3600s", dtypes={}
        )
        .reset_index()["timestamp"]
        .reset_index(drop=True)
    )
    gdf = DataFrame({"timestamp": timestamp.astype(dtype)})
    assert_eq(gdf, DataFrame.from_arrow(gdf.to_arrow(preserve_index=False)))


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([], dtype="datetime64[ns]"),
        pd.Series(pd.date_range("2010-01-01", "2010-02-01")),
        pd.Series([None, None], dtype="datetime64[ns]"),
    ],
)
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_datetime_unique(data, nulls):
    rng = np.random.default_rng(seed=0)
    psr = data.copy()

    if len(data) > 0:
        if nulls == "some":
            p = rng.integers(0, len(data), 2)
            psr[p] = None

    gsr = cudf.from_pandas(psr)
    expected = psr.unique()
    got = gsr.unique()

    # Unique does not provide a guarantee on ordering.
    assert_eq(
        pd.Series(expected).sort_values(ignore_index=True),
        got.sort_values(ignore_index=True).to_pandas(),
    )


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([], dtype="datetime64[ns]"),
        pd.Series(pd.date_range("2010-01-01", "2010-02-01")),
        pd.Series([None, None], dtype="datetime64[ns]"),
    ],
)
@pytest.mark.parametrize("nulls", ["none", "some"])
def test_datetime_nunique(data, nulls):
    psr = data.copy()
    rng = np.random.default_rng(seed=0)

    if len(data) > 0:
        if nulls == "some":
            p = rng.integers(0, len(data), 2)
            psr[p] = None

    gsr = cudf.from_pandas(psr)
    expected = psr.nunique()
    got = gsr.nunique()
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data, expected",
    [
        [["2018-01-01", None, "2019-01-31", None, "2018-01-01"], True],
        [
            [
                "2018-01-01",
                "2018-01-02",
                "2019-01-31",
                "2018-03-01",
                "2018-01-01",
            ],
            False,
        ],
        [
            np.array(
                ["2018-01-01", None, "2019-12-30"], dtype="datetime64[ms]"
            ),
            True,
        ],
    ],
)
def test_datetime_has_null_test(data, expected):
    data = Series(data, dtype="datetime64[ms]")
    pd_data = data.to_pandas()
    count = pd_data.notna().value_counts()
    expected_count = 0
    if False in count.keys():
        expected_count = count[False]

    assert_eq(expected, data.has_nulls)
    assert_eq(expected_count, data.null_count)


def test_datetime_has_null_test_pyarrow():
    data = Series(
        pa.array(
            [0, np.iinfo("int64").min, np.iinfo("int64").max, None],
            type=pa.timestamp("ns"),
        )
    )
    expected = True
    expected_count = 1

    assert_eq(expected, data.has_nulls)
    assert_eq(expected_count, data.null_count)


def test_datetime_dataframe():
    data = {
        "timearray": np.array(
            [0, 1, None, 2, 20, None, 897], dtype="datetime64[ms]"
        )
    }
    gdf = cudf.DataFrame(data)
    pdf = pd.DataFrame(data)

    assert_eq(pdf, gdf)

    assert_eq(pdf.dropna(), gdf.dropna())

    assert_eq(pdf.isnull(), gdf.isnull())

    data = np.array([0, 1, None, 2, 20, None, 897], dtype="datetime64[ms]")
    gs = cudf.Series(data)
    ps = pd.Series(data)

    assert_eq(ps, gs)

    assert_eq(ps.dropna(), gs.dropna())

    assert_eq(ps.isnull(), gs.isnull())


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas.",
)
def test_datetime_can_cast_safely():
    sr = cudf.Series(
        ["1679-01-01", "2000-01-31", "2261-01-01"], dtype="datetime64[ms]"
    )
    assert sr._column.can_cast_safely(np.dtype("datetime64[ns]"))

    sr = cudf.Series(
        ["1677-01-01", "2000-01-31", "2263-01-01"], dtype="datetime64[ms]"
    )

    assert sr._column.can_cast_safely(np.dtype("datetime64[ns]")) is False


# Cudf autocasts unsupported time_units
@pytest.mark.parametrize(
    "dtype",
    ["datetime64[D]", "datetime64[W]", "datetime64[M]", "datetime64[Y]"],
)
def test_datetime_array_timeunit_cast(dtype):
    testdata = np.array(
        [
            np.datetime64("2016-11-20"),
            np.datetime64("2020-11-20"),
            np.datetime64("2019-11-20"),
            np.datetime64("1918-11-20"),
            np.datetime64("2118-11-20"),
        ],
        dtype=dtype,
    )

    gs = Series(testdata)
    ps = pd.Series(testdata)

    assert_eq(ps, gs)

    gdf = DataFrame()
    gdf["a"] = np.arange(5)
    gdf["b"] = testdata

    pdf = pd.DataFrame()
    pdf["a"] = np.arange(5)
    pdf["b"] = testdata
    assert_eq(pdf, gdf)


@pytest.mark.parametrize("timeunit", ["D", "W", "M", "Y"])
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_datetime_scalar_timeunit_cast(timeunit):
    testscalar = np.datetime64("2016-11-20", timeunit)

    gs = Series(testscalar)
    ps = pd.Series(testscalar)

    assert_eq(ps, gs, check_dtype=False)

    gdf = DataFrame()
    gdf["a"] = np.arange(5)
    gdf["b"] = testscalar

    pdf = pd.DataFrame()
    pdf["a"] = np.arange(5)
    pdf["b"] = testscalar

    assert gdf["b"].dtype == cudf.dtype("datetime64[s]")
    assert_eq(pdf, gdf, check_dtype=True)


@pytest.mark.parametrize(
    "data",
    [
        ["2001-01-01", "2002-02-02", "2000-01-05", "NaT"],
        ["2001-01-01", "2002-02-02", "2000-01-05", None],
        [None, None, None, None, None],
    ],
)
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
def test_str_null_to_datetime(data, dtype):
    psr = pd.Series(data)
    gsr = Series(data)

    assert_eq(psr.astype(dtype), gsr.astype(dtype))


def test_str_to_datetime_error():
    psr = pd.Series(["2001-01-01", "2002-02-02", "2000-01-05", "None"])
    gsr = Series(["2001-01-01", "2002-02-02", "2000-01-05", "None"])

    assert_exceptions_equal(
        lfunc=psr.astype,
        rfunc=gsr.astype,
        lfunc_args_and_kwargs=(["datetime64[s]"],),
        rfunc_args_and_kwargs=(["datetime64[s]"],),
        check_exception_type=False,
    )


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4, 10, 100, 20000],
        [None] * 7,
        [10, 20, 30, None, 100, 200, None],
        [3223.234, 342.2332, 23423.23, 3343.23324, 23432.2323, 242.23, 233],
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [1, 2, 3, 4, 10, 100, 20000],
        [None] * 7,
        [10, 20, 30, None, 100, 200, None],
        [3223.234, 342.2332, 23423.23, 3343.23324, 23432.2323, 242.23, 233],
        datetime.datetime(1993, 6, 22, 13, 30),
        datetime.datetime(2005, 1, 22, 10, 00),
        np.datetime64("2005-02"),
        np.datetime64("2005-02-25"),
        np.datetime64("2005-02-25T03:30"),
        np.datetime64("nat"),
        # TODO: https://github.com/pandas-dev/pandas/issues/52295
    ],
)
@pytest.mark.parametrize("data_dtype", DATETIME_TYPES)
@pytest.mark.parametrize("other_dtype", DATETIME_TYPES)
def test_datetime_subtract(data, other, data_dtype, other_dtype):
    gsr = cudf.Series(data, dtype=data_dtype)
    psr = gsr.to_pandas()

    if isinstance(other, np.datetime64):
        gsr_other = other
        psr_other = other
    else:
        gsr_other = cudf.Series(other, dtype=other_dtype)
        psr_other = gsr_other.to_pandas()

    expected = psr - psr_other
    actual = gsr - gsr_other

    assert_eq(expected, actual)

    expected = psr_other - psr
    actual = gsr_other - gsr

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize(
    "other_scalars",
    [
        datetime.timedelta(days=768),
        datetime.timedelta(seconds=768),
        datetime.timedelta(microseconds=7),
        datetime.timedelta(minutes=447),
        datetime.timedelta(hours=447),
        datetime.timedelta(weeks=734),
        np.timedelta64(4, "s"),
        np.timedelta64(456, "D"),
        np.timedelta64(46, "h"),
        np.timedelta64("nat"),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64(1, "ns"),
    ],
)
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
@pytest.mark.parametrize(
    "op",
    ["add", "sub"],
)
def test_datetime_series_ops_with_scalars(data, other_scalars, dtype, op):
    gsr = cudf.Series(data=data, dtype=dtype)
    psr = gsr.to_pandas()

    if op == "add":
        expected = psr + other_scalars
        actual = gsr + other_scalars
    elif op == "sub":
        expected = psr - other_scalars
        actual = gsr - other_scalars

    assert_eq(expected, actual)

    if op == "add":
        expected = other_scalars + psr
        actual = other_scalars + gsr

        assert_eq(expected, actual)

    elif op == "sub":
        assert_exceptions_equal(
            lfunc=operator.sub,
            rfunc=operator.sub,
            lfunc_args_and_kwargs=([other_scalars, psr],),
            rfunc_args_and_kwargs=([other_scalars, gsr],),
        )


@pytest.mark.parametrize("data", ["20110101", "20120101", "20130101"])
@pytest.mark.parametrize("other_scalars", ["20110101", "20120101", "20130101"])
@pytest.mark.parametrize(
    "dtype",
    ["datetime64[ns]", "datetime64[us]", "datetime64[ms]", "datetime64[s]"],
)
def test_datetime_series_cmpops_with_scalars(data, other_scalars, dtype, op):
    gsr = cudf.Series(data=data, dtype=dtype)
    psr = gsr.to_pandas()

    expect = op(psr, other_scalars)
    got = op(gsr, other_scalars)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize(
    "scalar",
    [
        datetime.timedelta(days=768),
        datetime.timedelta(seconds=768),
        datetime.timedelta(microseconds=7),
        np.timedelta64("nat"),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64(1, "ns"),
    ],
)
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
@pytest.mark.parametrize("op", [np.add, np.subtract])
def test_datetime_series_ops_with_scalars_misc(data, scalar, dtype, op):
    gsr = cudf.Series(data=data, dtype=dtype)
    psr = gsr.to_pandas()

    expect = op(psr, scalar)
    got = op(gsr, scalar)

    assert_eq(expect, got)


def test_datetime_invalid_ops():
    sr = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    psr = sr.to_pandas()

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, pd.Timestamp(1513393355.5, unit="s")],),
        rfunc_args_and_kwargs=([sr, pd.Timestamp(1513393355.5, unit="s")],),
    )

    assert_exceptions_equal(
        lfunc=operator.truediv,
        rfunc=operator.truediv,
        lfunc_args_and_kwargs=([psr, pd.Timestamp(1513393355.5, unit="s")],),
        rfunc_args_and_kwargs=([sr, pd.Timestamp(1513393355.5, unit="s")],),
    )

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, psr],),
        rfunc_args_and_kwargs=([sr, sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.floordiv,
        rfunc=operator.floordiv,
        lfunc_args_and_kwargs=([psr, psr],),
        rfunc_args_and_kwargs=([sr, sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.floordiv,
        rfunc=operator.floordiv,
        lfunc_args_and_kwargs=([psr, pd.Timestamp(1513393355.5, unit="s")],),
        rfunc_args_and_kwargs=([sr, pd.Timestamp(1513393355.5, unit="s")],),
    )

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, 1],),
        rfunc_args_and_kwargs=([sr, 1],),
    )

    assert_exceptions_equal(
        lfunc=operator.truediv,
        rfunc=operator.truediv,
        lfunc_args_and_kwargs=([psr, "a"],),
        rfunc_args_and_kwargs=([sr, "a"],),
    )

    assert_exceptions_equal(
        lfunc=operator.mul,
        rfunc=operator.mul,
        lfunc_args_and_kwargs=([psr, 1],),
        rfunc_args_and_kwargs=([sr, 1],),
    )


@pytest.mark.parametrize(
    "data",
    [
        [],
        [1, 2, 3],
        [None, 1, 10, 11, None],
        [None, None, None, None, None],
        [None],
    ],
)
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
@pytest.mark.parametrize(
    "fill_value",
    [
        np.datetime64("2005-02"),
        np.datetime64("2005-02-25"),
        np.datetime64("2005-02-25T03:30"),
        np.datetime64("nat"),
        "NaT",
    ],
)
def test_datetime_fillna(data, dtype, fill_value):
    sr = cudf.Series(data, dtype=dtype)
    psr = sr.to_pandas()

    expected = psr.dropna()
    actual = sr.dropna()

    assert_eq(expected, actual)

    expected = psr.fillna(fill_value)
    actual = sr.fillna(fill_value)

    assert_eq(expected, actual)

    expected = expected.dropna()
    actual = actual.dropna()

    assert_eq(expected, actual)


@pytest.mark.parametrize("data", [[1, 2, 3], [], [1, 20, 1000, None]])
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
@pytest.mark.parametrize("stat", ["mean", "quantile"])
def test_datetime_stats(data, dtype, stat):
    gsr = cudf.Series(data, dtype=dtype)
    psr = gsr.to_pandas()

    with expect_warning_if(
        PANDAS_GE_230
        and stat == "quantile"
        and len(data) == 0
        and dtype != "datetime64[ns]"
    ):
        expected = getattr(psr, stat)()
    actual = getattr(gsr, stat)()

    if len(data) == 0:
        assert np.isnat(expected.to_numpy()) and np.isnat(actual.to_numpy())
    else:
        assert_eq(expected, actual)


@pytest.mark.parametrize("op", ["max", "min", "std", "median"])
@pytest.mark.parametrize(
    "data",
    [
        [],
        [1, 2, 3, 100],
        [10, None, 100, None, None],
        [None, None, None],
        [1231],
    ],
)
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
def test_datetime_reductions(data, op, dtype):
    sr = cudf.Series(data, dtype=dtype)
    psr = sr.to_pandas()

    actual = getattr(sr, op)()
    with expect_warning_if(
        psr.size > 0 and psr.isnull().all() and op == "median", RuntimeWarning
    ):
        expected = getattr(psr, op)()

    if (
        expected is pd.NaT
        and actual is pd.NaT
        or (np.isnat(expected.to_numpy()) and np.isnat(actual))
    ):
        assert True
    else:
        assert_eq(expected, actual)


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
@pytest.mark.parametrize("dtype", DATETIME_TYPES)
def test_datetime_infer_format(data, timezone, dtype):
    ts_data = [data + timezone]
    sr = cudf.Series(ts_data)
    psr = pd.Series(ts_data)
    if not timezone:
        expected = psr.astype(dtype)
        actual = sr.astype(dtype)

        assert_eq(expected, actual)
    else:
        with cudf.option_context("mode.pandas_compatible", True):
            with pytest.raises(NotImplementedError):
                # pandas doesn't allow parsing "Z" to naive type
                sr.astype(dtype)


def test_dateoffset_instance_subclass_check():
    assert not issubclass(pd.DateOffset, cudf.DateOffset)
    assert not isinstance(pd.DateOffset(), cudf.DateOffset)


def test_error_values():
    s = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    with pytest.raises(NotImplementedError, match="cupy does not support"):
        s.values


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize(
    "idx",
    [
        pd.DatetimeIndex([]),
        pd.DatetimeIndex(["2010-05-31"]),
        pd.date_range("2000-01-01", "2000-12-31", periods=21),
    ],
)
@pytest.mark.parametrize(
    "offset",
    [
        "10Y",
        "6M",
        "M",
        "31D",
        "0H",
        "44640T",
        "44640min",
        "2678000S",
        "2678000000L",
        "2678000000ms",
        "2678000000000U",
        "2678000000000us",
        "2678000000000000N",
        "2678000000000000ns",
    ],
)
def test_first(idx, offset):
    p = pd.Series(range(len(idx)), dtype="int64", index=idx)
    g = cudf.from_pandas(p)

    with pytest.warns(FutureWarning):
        expect = p.first(offset=offset)
    with pytest.warns(FutureWarning):
        got = g.first(offset=offset)

    assert_eq(expect, got)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
def test_first_start_at_end_of_month():
    idx = pd.DatetimeIndex(
        [
            "2020-01-31",
            "2020-02-15",
            "2020-02-29",
            "2020-03-15",
            "2020-03-31",
            "2020-04-15",
            "2020-04-30",
        ]
    )
    offset = "3M"
    p = pd.Series(range(len(idx)), index=idx)
    g = cudf.from_pandas(p)

    with pytest.warns(FutureWarning):
        expect = p.first(offset=offset)
    with pytest.warns(FutureWarning):
        got = g.first(offset=offset)

    assert_eq(expect, got)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="warning not present in older pandas versions",
)
@pytest.mark.parametrize(
    "idx",
    [
        pd.DatetimeIndex([]),
        pd.DatetimeIndex(["2010-05-31"]),
        pd.date_range("2000-01-01", "2000-12-31", periods=21),
    ],
)
@pytest.mark.parametrize(
    "offset",
    [
        "10Y",
        "6M",
        "M",
        "31D",
        "0H",
        "44640T",
        "44640min",
        "2678000S",
        "2678000000L",
        "2678000000ms",
        "2678000000000U",
        "2678000000000us",
        "2678000000000000N",
        "2678000000000000ns",
    ],
)
def test_last(idx, offset):
    p = pd.Series(range(len(idx)), dtype="int64", index=idx)
    g = cudf.from_pandas(p)

    with pytest.warns(FutureWarning):
        expect = p.last(offset=offset)
    with pytest.warns(FutureWarning):
        got = g.last(offset=offset)

    assert_eq(expect, got)


def test_datetime_binop_tz_timestamp(op):
    s = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    pd_tz_timestamp = pd.Timestamp("1970-01-01 00:00:00.000000001", tz="utc")
    with pytest.raises(NotImplementedError):
        op(s, pd_tz_timestamp)

    date_scalar = datetime.datetime.now(datetime.timezone.utc)
    with pytest.raises(NotImplementedError):
        op(s, date_scalar)


def test_datetime_series_cmpops_pandas_compatibility(op):
    data1 = ["20110101", "20120101", None, "20140101", None]
    data2 = ["20110101", "20120101", "20130101", None, None]
    gsr1 = cudf.Series(data=data1, dtype="datetime64[ns]")
    psr1 = gsr1.to_pandas()

    gsr2 = cudf.Series(data=data2, dtype="datetime64[ns]")
    psr2 = gsr2.to_pandas()

    expect = op(psr1, psr2)
    with cudf.option_context("mode.pandas_compatible", True):
        got = op(gsr1, gsr2)

    assert_eq(expect, got)


def test_datetime_getitem_na():
    s = cudf.Series([1, 2, None, 3], dtype="datetime64[ns]")
    assert s[2] is cudf.NaT


@pytest.mark.parametrize(
    "data, dtype",
    [
        [
            [
                "2000-01-01 00:00:00.000000000",
                "2000-01-01 00:00:00.000000000",
                "2000-01-01 00:00:00.000000000",
            ],
            "datetime64[s]",
        ],
        [
            [
                "2000-01-01 00:00:00.000000000",
                None,
                "2000-01-01 00:00:00.000000000",
            ],
            "datetime64[s]",
        ],
        [
            [
                "2000-01-01 00:00:00.001000000",
                "2000-01-01 00:00:00.000000000",
                "2000-01-01 00:00:00.000000000",
            ],
            "datetime64[us]",
        ],
        [
            [
                "2000-01-01 00:00:00.010000000",
                "2000-01-01 00:00:00.020000000",
                "2000-01-01 00:00:00.030000000",
            ],
            "datetime64[ms]",
        ],
        [
            [
                "2000-01-01 00:00:00.010000000",
                "2000-01-01 00:00:00.020000000",
                None,
            ],
            "datetime64[ms]",
        ],
        [
            [
                "2000-01-01 00:00:00.000001000",
                "2000-01-01 00:00:00.000000000",
                "2000-01-01 00:00:00.000004000",
            ],
            "datetime64[us]",
        ],
        [
            [
                None,
                "2000-01-01 00:00:00.000000000",
                "2000-01-01 00:00:00.000004000",
            ],
            "datetime64[us]",
        ],
        [
            [
                "2000-01-01 00:00:00.000000010",
                "2000-01-01 00:00:00.000000002",
                "2000-01-01 00:00:00.000000000",
            ],
            "datetime64[ns]",
        ],
        [
            [
                "2000-01-01 00:00:00.000000010",
                None,
                "2000-01-01 00:00:00.000000000",
            ],
            "datetime64[ns]",
        ],
        [
            [
                "2000-01-01 00:00:01.000000000",
                "2000-01-01 00:00:40.000000000",
                "2000-01-01 00:00:59.000000000",
            ],
            "datetime64[s]",
        ],
        [
            [
                "2000-01-01 00:10:00.000000000",
                "2000-01-01 00:30:40.000000000",
                "2000-01-01 00:59:00.000000000",
            ],
            "datetime64[s]",
        ],
        [
            [
                "2000-01-01 07:00:00.000000000",
                "2000-01-01 08:00:00.000000000",
                None,
            ],
            "datetime64[s]",
        ],
        [[None, None, None], "datetime64[s]"],
        [[], "datetime64[s]"],
        [
            [
                "2000-01-01 00:10:00.123456789",
                "2000-01-01 00:30:40.123123456",
                "2000-01-01 00:59:00.675347634",
            ],
            "datetime64[ns]",
        ],
    ],
)
def test_datetime_to_str(data, dtype):
    gs = cudf.Series(data, dtype=dtype)
    ps = gs.to_pandas()

    with cudf.option_context("mode.pandas_compatible", True):
        actual = gs.astype("str")

    expected = ps.astype("string")

    assert_eq(actual.to_pandas(nullable=True), expected)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_datetime_string_to_datetime_resolution_loss_raises():
    data = ["2020-01-01 00:00:00.00001"]
    dtype = "datetime64[s]"
    with pytest.raises(ValueError):
        cudf.Series(data, dtype=dtype)
    with pytest.raises(ValueError):
        pd.Series(data, dtype=dtype)


def test_timezone_pyarrow_array():
    pa_array = pa.array(
        [datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)],
        type=pa.timestamp("ns", "UTC"),
    )
    result = cudf.Series(pa_array)
    expected = pa_array.to_pandas()
    assert_eq(result, expected)


def test_dti_asi8():
    pd_dti = pd.DatetimeIndex(["2020-01-01", "2020-12-31"], name="foo")
    cudf_dti = cudf.from_pandas(pd_dti)

    result = pd_dti.asi8
    expected = cudf_dti.asi8
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "method, kwargs",
    [["mean", {}], ["std", {}], ["std", {"ddof": 0}]],
)
def test_dti_reduction(method, kwargs):
    pd_dti = pd.DatetimeIndex(["2020-01-01", "2020-12-31"], name="foo")
    cudf_dti = cudf.from_pandas(pd_dti)

    result = getattr(cudf_dti, method)(**kwargs)
    expected = getattr(pd_dti, method)(**kwargs)
    assert result == expected
