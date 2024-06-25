# Copyright (c) 2022-2024, NVIDIA CORPORATION.
import zoneinfo

import pandas as pd

import cudf
from cudf.testing import assert_eq


def test_tz_localize():
    tz = zoneinfo.ZoneInfo("America/New_York")
    pidx = pd.date_range("2001-01-01", "2001-01-02", freq="1s")
    pidx = pidx.astype("<M8[ns]")
    idx = cudf.from_pandas(pidx)
    assert pidx.dtype == idx.dtype
    assert_eq(pidx.tz_localize(tz), idx.tz_localize(tz))


def test_tz_convert():
    tz = zoneinfo.ZoneInfo("America/New_York")
    pidx = pd.date_range("2023-01-01", periods=3, freq="h")
    idx = cudf.from_pandas(pidx)
    pidx = pidx.tz_localize("UTC")
    idx = idx.tz_localize("UTC")
    assert_eq(pidx.tz_convert(tz), idx.tz_convert(tz))


def test_delocalize_naive():
    pidx = pd.date_range("2023-01-01", periods=3, freq="h")
    idx = cudf.from_pandas(pidx)
    assert_eq(pidx.tz_localize(None), idx.tz_localize(None))
