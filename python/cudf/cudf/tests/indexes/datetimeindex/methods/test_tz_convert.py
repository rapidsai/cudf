# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import zoneinfo

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_tz_convert():
    tz = zoneinfo.ZoneInfo("America/New_York")
    pidx = pd.date_range("2023-01-01", periods=3, freq="h")
    idx = cudf.from_pandas(pidx)
    pidx = pidx.tz_localize("UTC")
    idx = idx.tz_localize("UTC")
    assert_eq(pidx.tz_convert(tz), idx.tz_convert(tz))


def test_tz_convert_naive_typeerror():
    with pytest.raises(TypeError, match="Cannot convert tz-naive timestamps"):
        cudf.date_range("2020", periods=2, freq="D").tz_convert(None)
