# Copyright (c) 2023, NVIDIA CORPORATION.

import os

import pandas as pd
import pytest

import cudf
from cudf import date_range
from cudf.testing._utils import assert_eq


def _get_all_zones():
    zones = []
    for root, dirs, files in os.walk("/usr/share/zoneinfo"):
        for f in files:
            zone_name = ("/".join([root, f])).lstrip("/usr/share/zoneinfo")
            try:
                _ = pd.DatetimeTZDtype("ns", zone_name)
            except Exception:
                continue
            zones.append(zone_name)
    return zones


# NOTE: ALL_TIME_ZONES is a very large list; we likely do NOT want to
# use it for more than a handful of tests
ALL_TIME_ZONES = _get_all_zones()


@pytest.fixture(params=["ns", "us", "ms", "s"])
def unit(request):
    return request.param


@pytest.fixture(
    params=["America/New_York", "Asia/Tokyo", "CET", "Etc/GMT+1", "UTC"]
)
def tz(request):
    return request.param


@pytest.mark.parametrize("zone_name", ALL_TIME_ZONES)
def test_tz_localize(unit, zone_name):
    s = cudf.Series(date_range("2001-01-01", "2001-01-02", freq="1s"))
    s = s.astype(f"<M8[{unit}]")
    s = s.dt.tz_localize(zone_name)
    assert isinstance(s.dtype, pd.DatetimeTZDtype)
    assert s.dtype.unit == unit
    assert str(s.dtype.tz) == zone_name


@pytest.mark.parametrize("zone_name", ALL_TIME_ZONES)
def test_localize_ambiguous(request, unit, zone_name):
    request.applymarker(
        pytest.mark.xfail(
            condition=(zone_name == "America/Metlakatla"),
            reason="https://www.timeanddate.com/news/time/metlakatla-quits-dst.html",  # noqa: E501
        )
    )
    s = cudf.Series(
        [
            "2018-11-04 00:30:00",
            "2018-11-04 01:00:00",
            "2018-11-04 01:30:00",
            "2018-11-04 02:00:00",
            None,
            "2018-11-04 02:30:00",
        ],
        dtype=f"datetime64[{unit}]",
    )
    expect = s.to_pandas().dt.tz_localize(
        zone_name, ambiguous="NaT", nonexistent="NaT"
    )
    got = s.dt.tz_localize(zone_name)
    assert_eq(expect, got)


@pytest.mark.parametrize("zone_name", ALL_TIME_ZONES)
def test_localize_nonexistent(request, unit, zone_name):
    request.applymarker(
        pytest.mark.xfail(
            condition=(zone_name == "America/Grand_Turk"),
            reason="https://www.worldtimezone.com/dst_news/dst_news_turkscaicos03.html",  # noqa: E501
        )
    )
    s = cudf.Series(
        [
            "2018-03-11 01:30:00",
            "2018-03-11 02:00:00",
            "2018-03-11 02:30:00",
            "2018-03-11 03:00:00",
            None,
            "2018-03-11 03:30:00",
        ],
        dtype=f"datetime64[{unit}]",
    )
    expect = s.to_pandas().dt.tz_localize(
        zone_name, ambiguous="NaT", nonexistent="NaT"
    )
    got = s.dt.tz_localize(zone_name)
    assert_eq(expect, got)


def test_delocalize(unit, tz):
    psr = pd.Series(
        pd.date_range("2001-01-01", "2001-01-02", freq="1s")
    ).astype(f"datetime64[{unit}]")
    sr = cudf.from_pandas(psr)

    expect = psr.dt.tz_localize(tz).dt.tz_localize(None)
    got = sr.dt.tz_localize(tz).dt.tz_localize(None)
    assert_eq(expect, got)


def test_delocalize_naive():
    # delocalizing naive datetimes should be a no-op
    psr = pd.Series(["2001-01-01"], dtype="datetime64[ns]")
    sr = cudf.from_pandas(psr)

    expect = psr.dt.tz_localize(None)
    got = sr.dt.tz_localize(None)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "from_tz", ["Europe/London", "America/Chicago", "UTC"]
)
@pytest.mark.parametrize(
    "to_tz", ["Europe/London", "America/Chicago", "UTC", None]
)
def test_convert(from_tz, to_tz):
    ps = pd.Series(pd.date_range("2023-01-01", periods=3, freq="H"))
    gs = cudf.from_pandas(ps)
    ps = ps.dt.tz_localize(from_tz)
    gs = gs.dt.tz_localize(from_tz)
    expect = ps.dt.tz_convert(to_tz)
    got = gs.dt.tz_convert(to_tz)
    assert_eq(expect, got)


def test_convert_from_naive():
    gs = cudf.Series(cudf.date_range("2023-01-01", periods=3, freq="H"))
    with pytest.raises(TypeError):
        gs.dt.tz_convert("America/New_York")


@pytest.mark.parametrize(
    "data,original_timezone,target_timezone",
    [
        # DST transition:
        (["2023-03-12 01:30:00"], "America/New_York", "America/Los_Angeles"),
        # crossing the international date line:
        (["2023-05-17 23:30:00"], "Pacific/Auckland", "America/Los_Angeles"),
        # timezone with non-integer offset:
        (["2023-05-17 12:00:00"], "Asia/Kolkata", "Australia/Eucla"),
        # timezone with negative offset:
        (["2023-05-17 09:00:00"], "America/Los_Angeles", "Pacific/Auckland"),
        # conversion across multiple days:
        (["2023-05-16 23:30:00"], "America/New_York", "Asia/Kolkata"),
        # timezone with half-hour offset:
        (["2023-05-17 12:00:00"], "Asia/Kolkata", "Australia/Adelaide"),
        # timezone conversion with a timestamp in the future:
        (["2025-01-01 00:00:00"], "America/New_York", "Europe/London"),
        # timezone conversion with a timestamp in the past:
        (["2000-01-01 12:00:00"], "Europe/Paris", "America/Los_Angeles"),
        # timezone conversion with a timestamp at midnight:
        (["2023-05-17 00:00:00"], "Asia/Tokyo", "Europe/Paris"),
    ],
)
def test_convert_edge_cases(data, original_timezone, target_timezone):
    ps = pd.Series(data, dtype="datetime64[s]").dt.tz_localize(
        original_timezone
    )
    gs = cudf.Series(data, dtype="datetime64[s]").dt.tz_localize(
        original_timezone
    )
    expect = ps.dt.tz_convert(target_timezone)
    got = gs.dt.tz_convert(target_timezone)
    assert_eq(expect, got)


def test_to_pandas_index_true_timezone():
    data = [
        "2008-05-12",
        "2008-12-12",
        "2009-05-12",
    ]
    dti = cudf.DatetimeIndex(data).tz_localize("UTC")
    ser = cudf.Series(dti, index=list("abc"))
    result = ser.to_pandas(index=True)
    expected = pd.Series(pd.to_datetime(data, utc=True), index=list("abc"))
    assert_eq(result, expected)


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
