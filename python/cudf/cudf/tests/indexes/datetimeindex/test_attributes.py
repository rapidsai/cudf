# Copyright (c) 2025, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pandas._testing as tm
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "item, expected",
    [
        ["2020-01-01", False],
        ["2020-01-01T00:00:00+00:00", True],
        ["2020-01-01T00:00:00-08:00", False],
        ["2019-12-31T16:00:00-08:00", True],
    ],
)
def test_contains_tz_aware(item, expected):
    dti = cudf.date_range("2020", periods=2, freq="D").tz_localize("UTC")
    result = item in dti
    assert result == expected


@pytest.mark.parametrize("tz", ["UTC", None])
def test_tz_attribute(tz):
    dti = cudf.date_range("2020", periods=2, freq="D", tz=tz)
    if tz is None:
        assert dti.tz is None
    else:
        # TODO(pandas3.0-min): Assert zoneinfo.ZoneInfo(tz) == dti.tz
        assert str(dti.tz) == tz


def test_tz_aware_attributes_local():
    data = [
        "2008-05-12 13:50:00",
        "2008-12-12 14:50:35",
        "2009-05-12 13:50:32",
    ]
    dti = cudf.DatetimeIndex(data).tz_localize("UTC").tz_convert("US/Eastern")
    result = dti.hour
    expected = cudf.Index([9, 9, 9], dtype="int16")
    assert_eq(result, expected)


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
def test_dt_index(field):
    data = pd.DatetimeIndex(
        [pd.Timestamp(2020, 1, 2, 3, 4, 5, 6, nanosecond=7)]
    )
    gdf_data = cudf.DatetimeIndex(data)
    assert_eq(getattr(gdf_data, field), getattr(data, field), exact=False)


@pytest.mark.parametrize(
    "attr",
    [
        "is_month_start",
        "is_month_end",
        "is_quarter_end",
        "is_quarter_start",
        "is_year_end",
        "is_year_start",
        "days_in_month",
        "timetz",
        "time",
        "date",
    ],
)
def test_dti_datetime_attributes(attr):
    data = [
        "2020-01-01",
        "2020-01-31",
        "2020-03-01",
        "2020-03-31",
        "2020-03-31",
        "2020-12-31",
        None,
    ]
    pd_dti = pd.DatetimeIndex(data, name="foo")
    cudf_dti = cudf.from_pandas(pd_dti)

    result = getattr(cudf_dti, attr)
    expected = getattr(pd_dti, attr)
    if isinstance(result, np.ndarray):
        # numpy doesn't assert object arrays with NaT correctly
        tm.assert_numpy_array_equal(result, expected)
    else:
        assert_eq(result, expected)


@pytest.mark.parametrize("attr", ["freq", "unit"])
def test_dti_properties(attr):
    pd_dti = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"
    )
    cudf_dti = cudf.DatetimeIndex(
        ["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"
    )

    result = getattr(cudf_dti, attr)
    expected = getattr(pd_dti, attr)
    assert result == expected


def test_writable_numpy_array():
    gi = cudf.Index([1, 2, 3], dtype="datetime64[ns]")
    expected_flags = pd.Index(
        [1, 2, 3], dtype="datetime64[ns]"
    )._data._ndarray.flags

    actual_flags = gi.to_pandas()._data._ndarray.flags
    assert expected_flags.c_contiguous == actual_flags.c_contiguous
    assert expected_flags.f_contiguous == actual_flags.f_contiguous
    assert expected_flags.writeable == actual_flags.writeable
    assert expected_flags.aligned == actual_flags.aligned
    assert expected_flags.writebackifcopy == actual_flags.writebackifcopy


def test_dti_asi8():
    pd_dti = pd.DatetimeIndex(["2020-01-01", "2020-12-31"], name="foo")
    cudf_dti = cudf.from_pandas(pd_dti)

    result = pd_dti.asi8
    expected = cudf_dti.asi8
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "testlist",
    [
        [
            "2001-01-01 00:00:00",
            "2001-02-03 08:00:00",
            "2001-03-08 16:00:00",
            "2001-04-11 00:00:00",
        ],
        [
            "2001-04-11 00:00:00",
            "2001-03-08 16:00:00",
            "2001-02-03 08:00:00",
            "2001-01-01 00:00:00",
        ],
        [
            "2001-04-11 00:00:00",
            "2001-02-03 08:00:00",
            "2001-03-08 16:00:00",
            "2001-01-01 00:00:00",
        ],
        [
            "2001-04-11 00:00:00",
            "2001-01-01 00:00:00",
            "2001-02-03 08:00:00",
            "2001-03-08 16:00:00",
            "2001-01-01 00:00:00",
        ],
    ],
)
def test_datetime_index_is_unique_monotonic(testlist):
    index = cudf.DatetimeIndex(testlist)
    index_pd = pd.DatetimeIndex(testlist)

    assert index.is_unique == index_pd.is_unique
    assert index.is_monotonic_increasing == index_pd.is_monotonic_increasing
    assert index.is_monotonic_decreasing == index_pd.is_monotonic_decreasing
