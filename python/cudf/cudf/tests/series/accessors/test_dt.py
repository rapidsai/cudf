# Copyright (c) 2025, NVIDIA CORPORATION.

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import (
    PANDAS_CURRENT_SUPPORTED_VERSION,
    PANDAS_VERSION,
)
from cudf.testing import assert_eq


@pytest.fixture(
    params=[
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
        [
            136457654736252,
            134736784364431,
            245345345545332,
            223432411,
            2343241,
            3634548734,
            23234,
        ],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ]
)
def timedelta_data(request):
    return request.param


def test_timedelta_dt_components(timedelta_data, timedelta_types_as_str):
    gsr = cudf.Series(timedelta_data, dtype=timedelta_types_as_str)
    psr = gsr.to_pandas()

    expected = psr.dt.components
    actual = gsr.dt.components

    if gsr.isnull().any():
        assert_eq(expected, actual.astype("float"))
    else:
        assert_eq(expected, actual)


def test_timedelta_dt_properties(timedelta_data, timedelta_types_as_str):
    gsr = cudf.Series(timedelta_data, dtype=timedelta_types_as_str)
    psr = gsr.to_pandas()

    def local_assert(expected, actual, **kwargs):
        if gsr.isnull().any():
            assert_eq(expected, actual.astype("float"), **kwargs)
        else:
            assert_eq(expected, actual, **kwargs)

    expected_days = psr.dt.days
    actual_days = gsr.dt.days

    local_assert(expected_days, actual_days, check_dtype=False)

    expected_seconds = psr.dt.seconds
    actual_seconds = gsr.dt.seconds

    local_assert(expected_seconds, actual_seconds, check_dtype=False)

    expected_microseconds = psr.dt.microseconds
    actual_microseconds = gsr.dt.microseconds

    local_assert(expected_microseconds, actual_microseconds, check_dtype=False)

    expected_nanoseconds = psr.dt.nanoseconds
    actual_nanoseconds = gsr.dt.nanoseconds

    local_assert(expected_nanoseconds, actual_nanoseconds, check_dtype=False)


def test_timedelta_series_total_seconds(
    timedelta_data, timedelta_types_as_str
):
    gsr = cudf.Series(timedelta_data, dtype=timedelta_types_as_str)
    psr = gsr.to_pandas()

    expected = psr.dt.total_seconds()
    actual = gsr.dt.total_seconds()
    assert_eq(expected, actual)


@pytest.mark.parametrize("meth", ["day_name", "month_name"])
@pytest.mark.parametrize("klass", [cudf.Series, cudf.DatetimeIndex])
def test_day_month_name_locale_not_implemented(meth, klass):
    obj = klass(cudf.date_range("2020-01-01", periods=7))
    if klass is cudf.Series:
        obj = obj.dt
    with pytest.raises(NotImplementedError):
        getattr(obj, meth)(locale="pt_BR.utf8")


@pytest.mark.parametrize("meth", ["day_name", "month_name"])
@pytest.mark.parametrize("klass", [pd.Series, pd.DatetimeIndex])
def test_day_month_name(meth, klass):
    data = [
        "2020-05-31 08:00:00",
        None,
        "1999-12-31 18:40:00",
        "2000-12-31 04:00:00",
        None,
        "1900-02-28 07:00:00",
        "1800-03-14 07:30:00",
        "2100-03-14 07:30:00",
        "1970-01-01 00:00:00",
        "1969-12-31 12:59:00",
    ]

    p_obj = klass(data, dtype="datetime64[s]")
    g_obj = cudf.from_pandas(p_obj)

    if klass is pd.Series:
        p_obj = p_obj.dt
        g_obj = g_obj.dt

    expect = getattr(p_obj, meth)()
    got = getattr(g_obj, meth)()

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "freqstr",
    [
        "H",
        "N",
        "T",
        "L",
        "U",
        "S",
    ],
)
def test_datetime_ceil_raise_warning(freqstr):
    t = cudf.Series(
        ["2001-01-01 00:04:45", "2001-01-01 00:04:58", "2001-01-01 00:05:04"],
        dtype="datetime64[ns]",
    )
    with pytest.warns(FutureWarning):
        t.dt.ceil(freqstr)


@pytest.mark.parametrize(
    "resolution", ["D", "h", "min", "min", "s", "ms", "us", "ns"]
)
def test_round(datetime_types_as_str, resolution):
    data = [
        "2020-05-31 08:00:00",
        "1999-12-31 18:40:10",
        "2000-12-31 04:00:05",
        "1900-02-28 07:00:06",
        "1800-03-14 07:30:20",
        "2100-03-14 07:30:20",
        "1970-01-01 00:00:09",
        "1969-12-31 12:59:10",
    ]
    gs = cudf.Series(data, dtype=datetime_types_as_str)
    ps = gs.to_pandas()

    expect = ps.dt.round(resolution)
    got = gs.dt.round(resolution)
    assert_eq(expect, got)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="https://github.com/pandas-dev/pandas/issues/52761",
)
@pytest.mark.parametrize(
    "resolution", ["D", "h", "min", "min", "s", "ms", "us", "ns"]
)
def test_floor(datetime_types_as_str, resolution):
    data = [
        "2020-05-31 08:00:00",
        "1999-12-31 18:40:10",
        "2000-12-31 04:00:05",
        "1900-02-28 07:00:06",
        "1800-03-14 07:30:20",
        "2100-03-14 07:30:20",
        "1970-01-01 00:00:09",
        "1969-12-31 12:59:10",
    ]
    gs = cudf.Series(data, dtype=datetime_types_as_str)
    ps = gs.to_pandas()

    expect = ps.dt.floor(resolution)
    got = gs.dt.floor(resolution)
    assert_eq(expect, got)


@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="https://github.com/pandas-dev/pandas/issues/52761",
)
@pytest.mark.parametrize(
    "resolution", ["D", "h", "min", "min", "s", "ms", "us", "ns"]
)
def test_ceil(datetime_types_as_str, resolution):
    data = [
        "2020-05-31 08:00:00",
        "1999-12-31 18:40:10",
        "2000-12-31 04:00:05",
        "1900-02-28 07:00:06",
        "1800-03-14 07:30:20",
        "2100-03-14 07:30:20",
        "1970-01-01 00:00:09",
        "1969-12-31 12:59:10",
    ]
    gs = cudf.Series(data, dtype=datetime_types_as_str)
    ps = gs.to_pandas()

    expect = ps.dt.ceil(resolution)
    got = gs.dt.ceil(resolution)
    assert_eq(expect, got)


def test_days_in_months():
    data = [
        "2020-05-31",
        None,
        "1999-12-01",
        "2000-12-21",
        None,
        "1900-02-28",
        "1800-03-14",
        "2100-03-10",
        "1970-01-01",
        "1969-12-11",
    ]
    ps = pd.Series(data, dtype="datetime64[ns]")
    gs = cudf.from_pandas(ps)

    assert_eq(ps.dt.days_in_month, gs.dt.days_in_month)


def test_is_month_start():
    data = [
        "2020-05-31",
        None,
        "1999-12-01",
        "2000-12-21",
        None,
        "1900-02-28",
        "1800-03-14",
        "2100-03-10",
        "1970-01-01",
        "1969-12-11",
    ]
    ps = pd.Series(data, dtype="datetime64[ns]")
    gs = cudf.from_pandas(ps)

    expect = ps.dt.is_month_start
    got = gs.dt.is_month_start

    assert_eq(expect, got)


def test_is_month_end():
    data = [
        "2020-05-31",
        "2020-02-29",
        None,
        "1999-12-01",
        "2000-12-21",
        None,
        "1900-02-28",
        "1800-03-14",
        "2100-03-10",
        "1970-01-01",
        "1969-12-11",
    ]
    ps = pd.Series(data, dtype="datetime64[ns]")
    gs = cudf.from_pandas(ps)

    expect = ps.dt.is_month_end
    got = gs.dt.is_month_end

    assert_eq(expect, got)


def test_is_year_start():
    data = [
        "2020-05-31",
        None,
        "1999-12-01",
        "2000-12-21",
        None,
        "1900-01-01",
        "1800-03-14",
        "2100-03-10",
        "1970-01-01",
        "1969-12-11",
        "2017-12-30",
        "2017-12-31",
        "2018-01-01",
    ]
    ps = pd.Series(data, dtype="datetime64[ns]")
    gs = cudf.from_pandas(ps)

    expect = ps.dt.is_year_start
    got = gs.dt.is_year_start

    assert_eq(expect, got)


def test_is_year_end():
    data = [
        "2020-05-31",
        None,
        "1999-12-01",
        "2000-12-21",
        None,
        "1900-12-31",
        "1800-03-14",
        "2017-12-30",
        "2017-12-31",
        "2020-12-31 08:00:00",
        None,
        "1999-12-31 18:40:00",
        "2000-12-31 04:00:00",
        None,
        "1800-12-14 07:30:00",
        "2100-12-14 07:30:00",
        "2020-05-31",
    ]
    ps = pd.Series(data, dtype="datetime64[ns]")
    gs = cudf.from_pandas(ps)

    expect = ps.dt.is_year_end
    got = gs.dt.is_year_end

    assert_eq(expect, got)


def test_is_quarter_start():
    data = [
        "2020-05-01",
        "2020-05-31",
        "2020-02-29",
        None,
        "1999-12-01",
        "2000-12-21",
        None,
        "1900-02-28",
        "1800-03-14",
        "2100-03-10",
        "1970-04-1",
        "1970-01-01",
        "1969-12-11",
        "2020-12-31",
    ]
    ps = pd.Series(data, dtype="datetime64[ns]")
    gs = cudf.from_pandas(ps)

    expect = ps.dt.is_quarter_start
    got = gs.dt.is_quarter_start

    assert_eq(expect, got)


def test_is_quarter_end():
    data = [
        "2020-05-01",
        "2020-05-31",
        "2020-02-29",
        None,
        "1999-12-01",
        "2000-12-21",
        None,
        "1900-02-28",
        "1800-03-14",
        "2100-03-10",
        "1970-04-1",
        "1970-01-01",
        "1969-12-11",
        "2020-12-31",
    ]
    ps = pd.Series(data, dtype="datetime64[ns]")
    gs = cudf.from_pandas(ps)

    expect = ps.dt.is_quarter_end
    got = gs.dt.is_quarter_end

    assert_eq(expect, got)


def test_is_leap_year():
    data = [
        "2020-05-31 08:00:00",
        None,
        "1999-12-31 18:40:00",
        "2000-12-31 04:00:00",
        None,
        "1900-02-28 07:00:00",
        "1800-03-14 07:30:00",
        "2100-03-14 07:30:00",
        "1970-01-01 00:00:00",
        "1969-12-31 12:59:00",
    ]

    # Series
    ps = pd.Series(data, dtype="datetime64[s]")
    gs = cudf.from_pandas(ps)

    expect = ps.dt.is_leap_year
    got = gs.dt.is_leap_year

    assert_eq(expect, got)

    # DatetimeIndex
    pIndex = pd.DatetimeIndex(data)
    gIndex = cudf.from_pandas(pIndex)

    expect2 = pIndex.is_leap_year
    got2 = gIndex.is_leap_year

    assert_eq(expect2, got2)


def test_quarter():
    data = [
        "2020-05-31 08:00:00",
        "1999-12-31 18:40:00",
        "2000-12-31 04:00:00",
        "1900-02-28 07:00:00",
        "1800-03-14 07:30:00",
        "2100-03-14 07:30:00",
        "1970-01-01 00:00:00",
        "1969-12-31 12:59:00",
    ]
    dtype = "datetime64[s]"

    # Series
    ps = pd.Series(data, dtype=dtype)
    gs = cudf.from_pandas(ps)

    expect = ps.dt.quarter
    got = gs.dt.quarter

    assert_eq(expect, got, check_dtype=False)

    # DatetimeIndex
    pIndex = pd.DatetimeIndex(data)
    gIndex = cudf.from_pandas(pIndex)

    expect2 = pIndex.quarter
    got2 = gIndex.quarter

    assert_eq(expect2.values, got2.values)


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([], dtype="datetime64[ns]"),
        pd.Series(pd.date_range("2010-01-01", "2010-02-01")),
        pd.Series([None, None], dtype="datetime64[ns]"),
        pd.Series("2020-05-31 08:00:00", dtype="datetime64[s]"),
        pd.Series(
            pd.date_range(start="2021-07-25", end="2021-07-30"),
            index=["a", "b", "c", "d", "e", "f"],
        ),
    ],
)
def test_isocalendar_series(data):
    ps = data.copy()
    gs = cudf.from_pandas(ps)

    expect = ps.dt.isocalendar()
    got = gs.dt.isocalendar()

    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "data", [[1, 2, 3, None], [], [100121, 1221312, 321312321, 1232131223]]
)
@pytest.mark.parametrize(
    "date_format",
    [
        "%d - %m",
        "%y/%H",
        "%Y",
        "%I - %M / %S",
        "%f",
        "%j",
        "%p",
        "%w",
        "%U",
        "%W",
        "%G",
        "%u",
        "%V",
        "%b",
        "%B",
        "%a",
        "%A",
        "%U_",
        "_%b",
        "%B*",
        "%a ",
        "%A1",
    ],
)
def test_datetime_strftime(data, datetime_types_as_str, date_format):
    gsr = cudf.Series(data, dtype=datetime_types_as_str)
    psr = gsr.to_pandas()

    expected = psr.dt.strftime(date_format=date_format)
    actual = gsr.dt.strftime(date_format=date_format)

    assert_eq(expected, actual)


@pytest.mark.parametrize("date_format", ["%c", "%x", "%X"])
def test_datetime_strftime_not_implemented_formats(date_format):
    gsr = cudf.Series([1, 2, 3], dtype="datetime64[ms]")

    with pytest.raises(NotImplementedError):
        gsr.dt.strftime(date_format=date_format)


@pytest.mark.parametrize(
    "data",
    [
        pd.date_range("20010101", "20020215", freq="400h", name="times"),
        pd.date_range(
            "20010101", freq="243434324423423234ns", name="times", periods=10
        ),
    ],
)
@pytest.mark.parametrize(
    "field",
    [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "microsecond",
        "nanosecond",
        "weekday",
        "dayofweek",
        "dayofyear",
        "day_of_year",
    ],
)
def test_dt_series_datetime_fields(data, field):
    pd_data = pd.Series(data)
    gdf_data = cudf.Series(pd_data)
    base = getattr(pd_data.dt, field)
    test = getattr(gdf_data.dt, field)
    assert_eq(base, test, check_dtype=False)
