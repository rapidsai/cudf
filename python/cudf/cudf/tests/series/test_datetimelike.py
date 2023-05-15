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
