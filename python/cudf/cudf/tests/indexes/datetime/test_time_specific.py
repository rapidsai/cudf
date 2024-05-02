# Copyright (c) 2022-2024, NVIDIA CORPORATION.
import pandas as pd

import cudf
from cudf.testing._utils import assert_eq


def test_tz_localize():
    pidx = pd.date_range("2001-01-01", "2001-01-02", freq="1s")
    pidx = pidx.astype("<M8[ns]")
    idx = cudf.from_pandas(pidx)
    assert pidx.dtype == idx.dtype
    assert_eq(
        pidx.tz_localize("America/New_York"),
        idx.tz_localize("America/New_York"),
    )


def test_tz_convert():
    pidx = pd.date_range("2023-01-01", periods=3, freq="h")
    idx = cudf.from_pandas(pidx)
    pidx = pidx.tz_localize("UTC")
    idx = idx.tz_localize("UTC")
    assert_eq(
        pidx.tz_convert("America/New_York"), idx.tz_convert("America/New_York")
    )


def test_delocalize_naive():
    pidx = pd.date_range("2023-01-01", periods=3, freq="h")
    idx = cudf.from_pandas(pidx)
    assert_eq(pidx.tz_localize(None), idx.tz_localize(None))
