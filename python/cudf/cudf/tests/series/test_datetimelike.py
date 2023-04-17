# Copyright (c) 2023, NVIDIA CORPORATION.

import pytest

import cudf
from cudf import date_range
from cudf.core.dtypes import DatetimeTZDtype
from cudf.testing._utils import assert_eq


@pytest.fixture(params=["ns", "us", "ms", "s"])
def unit(request):
    return request.param


@pytest.fixture(params=["America/New_York", "Asia/Tokyo", "CET", "Etc/GMT+1"])
def tz(request):
    return request.param


def test_tz_localize(unit, tz):
    s = cudf.Series(date_range("2001-01-01", "2001-01-02", freq="1s"))
    s = s.astype(f"<M8[{unit}]")
    s = s.dt.tz_localize(tz)
    assert isinstance(s.dtype, DatetimeTZDtype)
    assert s.dtype.unit == unit
    assert str(s.dtype.tz) == tz


def test_localize_ambiguous(unit, tz):
    s = cudf.Series(
        [
            "2018-11-04 00:30:00",
            "2018-11-04 01:00:00",
            "2018-11-04 01:30:00",
            "2018-11-04 02:00:00",
            "2018-11-04 02:30:00",
        ],
        dtype=f"datetime64[{unit}]",
    )
    expect = s.to_pandas().dt.tz_localize(tz, ambiguous="NaT")
    got = s.dt.tz_localize(tz)
    assert_eq(expect, got)


def test_localize_nonexistent(unit, tz):
    s = cudf.Series(
        [
            "2018-03-11 01:30:00",
            "2018-03-11 02:00:00",
            "2018-03-11 02:30:00",
            "2018-03-11 03:00:00",
            "2018-03-11 03:30:00",
        ],
        dtype=f"datetime64[{unit}]",
    )
    expect = s.to_pandas().dt.tz_localize(tz, nonexistent="NaT")
    got = s.dt.tz_localize(tz)
    assert_eq(expect, got)
